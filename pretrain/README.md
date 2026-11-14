# Transformer 语言模型

八种小模型的 Scaling Law 实验配置与启动方法见 [SCALING.md](SCALING.md)。

`model.py` 实现 decoder-only Transformer。

## 模型配置

`ModelConfig` 的默认尺寸采用 `assignment/aml-hw1.pdf` 的参考配置：

| 配置 | 值 |
|---|---:|
| 词表大小 `vocab_size` | 16000 |
| 隐藏维度 `d_model` | 1024 |
| 层数 `n_layers` | 16 |
| 注意力头数 `n_heads` | 16 |
| 每头维度 `head_dim` | 64，由 `d_model / n_heads` 得到 |
| 前馈层维度 `d_ff` | 4096 |
| 上下文长度 `context_length` | 1024 |

按本次指定的架构使用 RoPE、GELU、RMSNorm、Pre-Norm 和独立输出头。
RoPE 的 `rope_theta=10000`、RMSNorm 的 `norm_eps=1e-6` 是实现默认值，PDF 未指定。
线性层均不使用 bias，不使用 dropout，也不共享输入嵌入与输出头的权重。
Embedding 和线性层按标准差 0.02 初始化，残差分支的输出投影标准差额外除以 `sqrt(2 * n_layers)`。

## 结构

```text
input_ids [B, T]
    ↓
Token Embedding → [B, T, 1024]
    ↓
16 × TransformerBlock:
    x = x + CausalSelfAttention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))
    ↓
Final RMSNorm
    ↓
独立 Linear 输出头 → logits [B, T, 16000]
```

注意力中，将 Q/K/V 划分为 16 个头，对 Q 和 K 的相邻坐标对应用 RoPE，
再计算 `softmax(QKᵀ / sqrt(head_dim) + causal_mask) V`。
RoPE 不加在 token embedding 上，也不作用于 V；位置从 0 开始。
因果掩码将主对角线上方设为负无穷，每个位置只能关注自身和之前的位置。
传入 `document_ids` 时，再屏蔽其他文档的位置，形成文档内因果注意力；同一个 mask 由所有层复用。
训练和验证均传入文档编号。RoPE 仍按块内位置计算，文档内相对位置保持不变。
FFN 为 `Linear(1024, 4096) → GELU → Linear(4096, 1024)`。
注意力、RoPE、RMSNorm 和残差连接的计算均在代码中显式实现。

参数量：

| 部分 | 参数量 |
|---|---:|
| Token embedding | 16,384,000 |
| 16 个 Transformer block | 201,359,360 |
| Final RMSNorm | 1,024 |
| 独立输出头 | 16,384,000 |
| 总计 | 234,128,384 |

PDF 标注总参数量约 218M。本实现保留独立输出头，因此总计约 234M；
共享 embedding 和输出头可减少 16,384,000 个参数，但这里按要求使用独立权重。

## 前向计算与损失

在项目根目录执行 Python 代码：

```python
import torch
from torch.nn import functional as F
from pretrain import ModelConfig, TransformerLM

model = TransformerLM(ModelConfig()).cuda()
pad_token_id = 3  # 参考分词器的 <pad> ID
tokens = torch.randint(4, 16000, (1, 33), device="cuda")
tokens[0, 0] = 1  # 参考分词器的 <s> ID
tokens[0, -4] = 2  # 参考分词器的 </s> ID
tokens[0, -3:] = pad_token_id

# 输入和目标错开一个 token，模型内部不再移动目标。
inputs, targets = tokens[:, :-1], tokens[:, 1:]
document_ids = (inputs == 1).long().cumsum(dim=-1)
targets = targets.clone()
targets[targets == 1] = pad_token_id  # BOS 仅作为起始上下文，不预测下一篇文档的 BOS
logits = model(inputs, document_ids=document_ids)  # [1, 32, 16000]
loss = F.cross_entropy(
    logits.reshape(-1, 16000), targets.reshape(-1), ignore_index=pad_token_id,
)
loss.backward()
print(logits.shape, loss.item())
```

`forward()` 返回未经 softmax 的 logits，训练时直接交给交叉熵。
输入是 `[batch, length]` 的整数 token ID，长度为 1 到 `context_length`。
当前接口处理等长序列，不接收 padding mask；后续预训练可使用打包后的等长 token 块。
上例采用右侧 PAD 补齐，因果掩码保证有效位置无法关注后面的 PAD。
交叉熵通过 `ignore_index=pad_token_id` 排除 PAD 和已屏蔽的 BOS 目标，保留正文与 EOS 的损失和梯度。
仅在输入中填入 PAD 并不会自动屏蔽 loss；`generate()` 本身在无梯度模式下运行。
`model.py` 不读取数据集，不自动添加特殊 token；训练由下面的第三部分代码完成。

## 自回归采样

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("flexitok/bpe_fw_edu_16000")
tokenizer.no_padding()
tokenizer.no_truncation()
prompt = tokenizer.encode("Once upon a time", add_special_tokens=False).ids
prompt = [tokenizer.token_to_id("<s>")] + prompt
input_ids = torch.tensor([prompt], device="cuda")

output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    eos_token_id=tokenizer.token_to_id("</s>"),
    pad_token_id=tokenizer.token_to_id("<pad>"),
)
print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))
```

此段接在上面的模型创建代码之后运行。随机初始化的模型只能用于检查采样流程；
生成连贯文本需要先完成第三部分的训练并加载训练权重。

`temperature=0` 使用贪心选择，正数使用温度采样；`top_k=None` 从完整词表采样。
返回结果包含输入提示词。指定 EOS 后，每行在首次生成 EOS 时完成，
已完成的行后续填充 PAD，直到整个 batch 完成或达到最大生成长度。
使用 EOS 停止时，必须传入与 EOS 不同的 `pad_token_id`；首次生成的 EOS 会保留。
指定 PAD 后，它不参与正常的 token 采样，只用于补齐已结束的行。
生成时自动切换到评估模式，并在结束后恢复原来的模式。
每步只使用最近 `context_length` 个 token，从位置 0 重新计算，不使用 KV cache。

## Part 3：TinyStories 预训练

`run.sh` 默认使用 TinyStories、参考分词器 `flexitok/bpe_fw_edu_16000`、2 张 GPU（2、3）、
每卡 batch 16、梯度累积 4、长度 1024，训练 1 个 epoch。
模型、AdamW、学习率计划和文档内因果注意力沿用前面的配置。

### 数据与验证

`prepare.py` 使用 TinyStories 的官方 train / validation 文件，不重新随机划分。
本机训练集 2,119,719 篇，验证集 21,990 篇，均完整编码。
数据目录为 `/user-data/chaojh26/TinyStories/`，编码目录为该目录下的 `packed-1024/`。
每篇文档保存为 `BOS + 正文 + EOS`，连续打包后按块随机打乱训练，块内 token 顺序保持不变。
注意力只关注本篇文档中当前位置及之前的 token；BOS 和 PAD 目标忽略，正文与 EOS 计入 loss。
每块保存原始 UTF-8 字节数，供 BPB 计算使用。训练块末尾不足一个全局 batch 的余量舍去。

训练前、每 100 步、最后一步均遍历**完整验证集**，没有验证子集或采样上限。
各 GPU 处理互不重复的验证块，汇总负对数似然、有效 token 数与原文字节数，计算 loss、PPL、BPB。
`best.pt` 按完整验证集 loss 选择。结束后再次加载最佳 checkpoint，生成最终指标、样例和曲线。
训练 loss 曲线来自实际训练 batch。

```bash
python -m pretrain.prepare \
    --dataset tinystories \
    --data-dir /user-data/chaojh26/TinyStories \
    --output-dir /user-data/chaojh26/TinyStories/packed-1024 \
    --tokenizer flexitok/bpe_fw_edu_16000 \
    --context-length 1024 --batch-size 1024
```

编码产物为 `train.bin`、`validation.bin`、对应的 `*.bytes.bin`、`metadata.json` 和 `tokenizer.json`。
脚本发现完整编码的 `metadata.json` 后跳过准备数据。准备过程不支持中途续写，未完成时需改用空目录。
`prepare.py` 仍支持 `--dataset fineweb`，此时需显式指定 FineWeb 原始目录和独立输出目录，按 seed 随机划分 99:1。

### 启动与日志

当前环境的 `wandb==0.30.0` 搭配已验证可访问的官方公共入口：

```bash
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_APP_URL=https://forge.coreweave.com/wandb
wandb login --host "$WANDB_BASE_URL" --verify
```

Key 从 [官方授权页](https://forge.coreweave.com/wandb/authorize) 获取，完整保留 `wandb_v1_` 前缀。
`run.sh` 已设置上述服务地址；项目为 `aml-hw1`，运行名为 `20261113-tinystories-1`。
在线模式在初始化多卡进程组前检查登录。可用 `--wandb-mode offline` 仅记录本地 W&B 数据，
或 `--wandb-mode disabled` 关闭 W&B；tqdm、本地指标和 checkpoint 仍会保存。

在项目根目录启动，并将输出连接到文件、脱离原终端会话：

```bash
nohup setsid bash pretrain/run.sh > pretrain/train-tinystories.log 2>&1 < /dev/null &
tail -f pretrain/train-tinystories.log
```

仅执行 `bg/disown` 不会把已有输出改为文件，关闭原终端可能使日志刷新失败。
日志中的 step 是优化器更新次数；131,072 为每步处理的目标位置数，实际有效 token 数排除 BOS 与 PAD。

### 训练产物与单独使用

默认脚本输出目录：`/user-data/chaojh26/pretrain/runs/20261113-tinystories-1/`。

| 文件 | 内容 |
|---|---|
| `config.json` | 模型、数据、训练配置与实际参数量 |
| `metrics.jsonl` | 训练和完整验证的 loss、PPL、BPB、步数、token 数、耗时等 |
| `last.pt` | 最近保存的模型、优化器、进度和各卡随机状态 |
| `best.pt` | 完整验证集 loss 最低的 checkpoint |
| `samples.jsonl` | 周期生成的样例 |
| `final_metrics.json`、`final_samples.json` | 最佳 checkpoint 的最终指标与生成样例 |
| `curves.png`、`curves.pdf` | 本地指标曲线 |
| `wandb/` | W&B 本地记录 |

验证后、每 1000 步及最后一步保存 `last.pt`，临时文件写完后替换；最优模型另存为 `best.pt`。
checkpoint 包含分词器 JSON，后续生成不需联网下载分词器。
恢复可使用原训练命令加 `--resume /路径/last.pt`，保持数据、模型、GPU 数、batch、累积和优化配置一致。
`run.sh` 使用固定参数，不转发额外参数。

```bash
python -m pretrain.evaluate \
    --checkpoint /user-data/chaojh26/pretrain/runs/20261113-tinystories-1/best.pt \
    --split validation --output-dir /user-data/chaojh26/pretrain/runs/20261113-tinystories-1/evaluation

python -m pretrain.infer \
    --checkpoint /user-data/chaojh26/pretrain/runs/20261113-tinystories-1/best.pt \
    --prompt "Once upon a time" --max-new-tokens 100 --temperature 0.8 --top-k 50

python -m pretrain.plot \
    --metrics /user-data/chaojh26/pretrain/runs/20261113-tinystories-1/metrics.jsonl \
    --output-dir /user-data/chaojh26/pretrain/runs/20261113-tinystories-1
```

文档隔离参考 [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/)。
当前使用显式掩码实现，没有使用论文中的变长 FlashAttention 内核，论文的效率数字不适用于这里。
