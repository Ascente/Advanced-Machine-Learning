# Transformer 语言模型

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

## Part 3：FineWeb 预训练

新增文件：

| 文件 | 作用 |
|---|---|
| `prepare.py` | 随机划分 FineWeb，用参考分词器编码并记录原始字节数 |
| `data.py` | 内存映射读取 token，构造等长输入、目标及文档编号 |
| `train.py` | 单卡或 DDP 多卡训练、验证、checkpoint、W&B 和 tqdm |
| `evaluate.py` | 对完整训练集或验证集计算 loss、PPL、BPB |
| `infer.py` | 从 checkpoint 加载模型和分词器，生成文本 |
| `plot.py` | 从本地日志绘制 loss、PPL、BPB 曲线 |
| `run.sh` | 数据准备后执行 2 卡主模型训练，显式列出参数 |

以下命令均在项目根目录执行。当前服务器已安装所需的 PyTorch、NumPy、
PyArrow、tokenizers、tqdm、wandb 和 matplotlib。

### 数据准备

```bash
python -m pretrain.prepare \
    --data-dir /user-data/chaojh26/FineWeb-Edu/sample-10BT \
    --output-dir /user-data/chaojh26/FineWeb-Edu/packed-1024 \
    --tokenizer flexitok/bpe_fw_edu_16000 \
    --context-length 1024 \
    --seed 42 \
    --batch-size 1024
```

按文件名和行序编号，使用与 `tokenizer/corpus.py` 相同的随机规则：
`random.Random(seed).sample(range(total), total // 100)` 抽取验证文档，其余为训练文档。
本机文档数为训练 9,575,380、验证 96,721。先划分文档，再分别拼接，两个 split 不混合。

采用论文 [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://aclanthology.org/2024.acl-long.427/)
中的文档隔离思路：仍将文档紧凑打包，但注意力只在文档内部计算。
本实现用显式布尔 mask 隔离文档；论文中的 FlashAttention 变长内核未引入，论文的效率数字不适用于本实现。
文档顺序沿用本地 Parquet 顺序，不进行 BM25 相关文档检索排序。

每篇文档按 `BOS + 正文 + EOS` 编码，使用参考分词器的 `<s>` 和 `</s>`。
网页中直接出现的 `<s>`、`</s>`、`<pad>` 等字符串按普通文本编码。
每个 split 写一个连续的 uint16 token 文件，在训练时以 1024 个目标 token 为一个块读取；
输入与目标错开一个位置，相邻块共享一个上下文 token，目标不重复。
根据输入中的 BOS 构造文档编号，只有编号相同且不晚于当前位置的 token 可以互相注意。
BOS 为本篇首个正文 token 提供起始上下文；正文和 EOS 参与 loss，目标中的 BOS 与 PAD 忽略。
这样不会训练“从上一篇文档的 EOS 预测下一篇的 BOS”这一跨文档目标。
最后一个不足长度的块在读取时右侧填 PAD。一篇长文档可以跨块，每个块独立计算。

输出：`train.bin`、`validation.bin`、对应的 `*.bytes.bin`、`tokenizer.json`、`metadata.json`。
`*.bytes.bin` 每个块仅存一个原始 UTF-8 字节数，而非逐 token 保存一份字节数组。
依据编码 offsets 对原文分段计数，重复字节 token 的字符只计一次，并保留 NFC 归一化前的原文字节数。
每个 split 的首 token 是仅作为上下文的 BOS，所有正文 token 均参与计分。
BOS、EOS 和 PAD 不增加字节数，BPB 分母包含完整的原始文本字节数。
BPB 为 `总负对数似然 / (ln(2) × 被计分文本的原始 UTF-8 字节数)`。

数据准备会遍历完整数据集，需要单独等待完成，只需执行一次。
它要求空输出目录，不支持从中断的编码进度续写；中断后可换一个空目录重跑。
训练阶段支持 checkpoint 恢复。更改上下文长度或数据划分 seed 后应准备新的数据目录。
旧的无 BOS 编码数据不能直接用于此训练流程，需要使用空输出目录重新编码。

### 训练配置与启动

主配置沿用第二部分的模型，独立输出头使总参数量为 234,128,384。
优化器 AdamW，`betas=(0.9, 0.95)`、`eps=1e-8`；二维权重使用 weight decay 0.1，
RMSNorm 缩放权重不衰减。学习率峰值 4e-4，前 2% 更新线性 warmup，
之后余弦衰减至 4e-5；梯度范数裁剪为 1.0，默认 BF16 混合精度。

2 卡配置每次更新处理 `2 卡 × 每卡 16 条 × 累积 4 次 × 1024 = 131,072` 个目标位置。
其中 BOS 和 PAD 位置不计入有效 token 数，日志中的 `tokens_seen` 为实际参与 loss 的正文与 EOS 数。
`--batch-size` 是每卡每次前向的序列数，`--gradient-accumulation` 是一次优化器更新累积的次数。
以实际未被忽略的目标数归一化梯度，多卡 DDP 只在最后一次累积时同步梯度。
默认遍历 1 个 epoch；每个 epoch 随机打乱训练块，舍去不足一个全局有效 batch 的余量。
`--max-steps` 可限制总更新次数，学习率计划也使用这个实际总步数。

先配置自己的 W&B 登录凭据：

```bash
export WANDB_BASE_URL=https://forge.coreweave.com/api/wandb
export WANDB_APP_URL=https://forge.coreweave.com/wandb
wandb login --host "$WANDB_BASE_URL" --verify

# 若数据尚未编码则先准备数据，再执行 2 卡训练
bash pretrain/run.sh
```

`run.sh` 默认使用 2 张 GPU，使用 `https://forge.coreweave.com/api/wandb` 访问 W&B 公共平台，
网页入口为 `https://forge.coreweave.com/wandb`。
当前服务器通过代理访问 API 子域名返回 403，而这个官方入口正常。
本环境使用已验证的 `wandb==0.30.0`；0.31.0 会将该入口改写为当前无法访问的 API 子域名。
API Key 从 [官方授权页](https://forge.coreweave.com/wandb/authorize) 获取，在终端登录时输入，
完整保留 `wandb_v1_` 前缀，不写入脚本。
已有编码数据的 `metadata.json` 时自动跳过数据准备。
在线模式会在创建多卡进程组之前检查 W&B 登录；未登录时直接报错，不在训练中等待输入。
请先按上面的命令登录公共平台，确保登录和训练使用同一个服务地址。
团队项目可增加 `--wandb-entity 团队名称`。
`--wandb-mode offline` 仅保存本地 W&B 数据，之后使用 `wandb sync 离线运行目录` 上传；
`--wandb-mode disabled` 关闭 W&B，仍保存本地指标并显示 tqdm。
只由 rank 0 记录 W&B 和本地文件，所有 rank 参与训练及验证。

单卡示例，保持同样的有效 batch：

```bash
python -m pretrain.train \
    --data-dir /user-data/chaojh26/FineWeb-Edu/packed-1024 \
    --output-dir /user-data/chaojh26/pretrain/fineweb-234m-single \
    --batch-size 16 --gradient-accumulation 8 \
    --epochs 1 --precision bf16 \
    --wandb-mode online --wandb-project aml-hw1
```

tqdm 显示优化器更新进度、loss、学习率和已训练 token 数，验证也有独立进度条。
每 10 次更新记录训练指标；每 250 次更新在固定的前 100 个全局验证 batch 上评估，
并生成三个提示词的样例。`--sample-tokens 0` 可关闭训练期间生成。
周期验证是固定子集，用于选择最佳 checkpoint，不应当作完整验证集结果。
训练结束后加载 `best.pt`，遍历全部验证块，输出最终指标、生成样例及本地曲线。
交叉熵按有效 token 数加权，PPL 为 `exp(loss)`，BPB 使用同一批被计分目标对应的原文字节数。

### 输出与恢复

默认训练目录为 `/user-data/chaojh26/pretrain/fineweb-234m/`：

| 文件 | 内容 |
|---|---|
| `config.json` | 模型、数据和训练配置，实际参数量与有效 batch |
| `metrics.jsonl` | 训练与验证的 loss、PPL、BPB、token 数、步数、耗时等 |
| `last.pt` | 最近一次保存的模型、优化器、进度及各 rank 的随机状态 |
| `best.pt` | 固定验证子集上 loss 最小的 checkpoint |
| `samples.jsonl` | 周期验证时的生成样例 |
| `final_metrics.json` | 最佳 checkpoint 在完整验证集上的指标 |
| `final_samples.json` | 最佳 checkpoint 的生成样例 |
| `curves.png`、`curves.pdf` | 训练与验证的 loss、PPL、BPB 曲线 |
| `wandb/` | W&B 本地运行记录（启用 W&B 时） |

`last.pt` 在验证时、每 1000 次更新时及最后一次更新后保存，使用临时文件替换。
`best.pt` 保留最优模型；`last.pt` 保留训练结束时的模型，即使它不是最优模型。
checkpoint 同时保存分词器 JSON，后续生成不需要再次联网下载。
日志还记录近似 `6 × 实际总参数量 × 已训练 token 数` FLOPs，供后续实验分析。
训练曲线使用实际训练 batch 的指标，不额外遍历整个训练集；最终完整验证结果单独标注。

恢复时使用原训练命令，加上 `--resume`，保持 GPU 数、batch、累积次数、模型和优化配置一致：

```bash
torchrun --standalone --nproc_per_node=2 -m pretrain.train \
    --data-dir /user-data/chaojh26/FineWeb-Edu/packed-1024 \
    --output-dir /user-data/chaojh26/pretrain/fineweb-234m \
    --resume /user-data/chaojh26/pretrain/fineweb-234m/last.pt \
    --batch-size 16 --gradient-accumulation 4 \
    --epochs 1 --precision bf16 \
    --wandb-mode online --wandb-project aml-hw1
```

恢复读取同一输出目录中的 checkpoint，按已完成更新数恢复 epoch 内的数据位置，
并恢复优化器、学习率进度和随机状态。在线 W&B 复用保存的 run ID。
恢复会删除本地指标中晚于该 checkpoint 的记录，避免重复的未来曲线。

### 独立评估、生成与绘图

```bash
python -m pretrain.evaluate \
    --checkpoint /user-data/chaojh26/pretrain/fineweb-234m/best.pt \
    --split validation \
    --output-dir /user-data/chaojh26/pretrain/fineweb-234m/evaluation

python -m pretrain.infer \
    --checkpoint /user-data/chaojh26/pretrain/fineweb-234m/best.pt \
    --prompt "The solar system consists of" \
    --max-new-tokens 100 --temperature 0.8 --top-k 50

python -m pretrain.plot \
    --metrics /user-data/chaojh26/pretrain/fineweb-234m/metrics.jsonl \
    --output-dir /user-data/chaojh26/pretrain/fineweb-234m
```

独立评估始终遍历整个所选 split，`--split train` 可评估完整训练集。
以上训练代码使用显式注意力实现；显存不足时可以减小每卡 batch，并相应增加梯度累积次数。
