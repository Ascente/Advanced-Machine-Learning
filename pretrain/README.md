# Transformer 语言模型

`model.py` 实现 decoder-only Transformer。

## 模型配置

`ModelConfig` 的默认尺寸采用 `homework/aml-hw1.pdf` 的参考配置：

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
tokens = torch.randint(0, 16000, (1, 33), device="cuda")

# 输入和目标错开一个 token，模型内部不再移动目标。
inputs, targets = tokens[:, :-1], tokens[:, 1:]
logits = model(inputs)  # [1, 32, 16000]
loss = F.cross_entropy(logits.reshape(-1, 16000), targets.reshape(-1))
loss.backward()
print(logits.shape, loss.item())
```

`forward()` 返回未经 softmax 的 logits，训练时直接交给交叉熵。
输入是 `[batch, length]` 的整数 token ID，长度为 1 到 `context_length`。
当前接口处理等长序列，不接收 padding mask；后续预训练可使用打包后的等长 token 块。
这一部分不读取数据集、不自动添加特殊 token，也不包含第三部分的训练循环。

## 自回归采样

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("flexitok/bpe_fw_edu_16000")
tokenizer.no_padding()
tokenizer.no_truncation()
prompt = tokenizer.encode("Once upon a time", add_special_tokens=False).ids
input_ids = torch.tensor([prompt], device="cuda")

output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    eos_token_id=tokenizer.token_to_id("</s>"),
)
print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))
```

此段接在上面的模型创建代码之后运行。随机初始化的模型只能用于检查采样流程；
生成连贯文本需要先完成第三部分的训练并加载训练权重。

`temperature=0` 使用贪心选择，正数使用温度采样；`top_k=None` 从完整词表采样。
返回结果包含输入提示词。指定 EOS 后，每行在首次生成 EOS 时完成，
已完成的行继续填充 EOS，直到整个 batch 完成或达到最大生成长度。
生成时自动切换到评估模式，并在结束后恢复原来的模式。
每步只使用最近 `context_length` 个 token，从位置 0 重新计算，不使用 KV cache。
