# BPE 分词器

自行实现的字节级 BPE，使用完整 TinyStories 训练集学习 16,000 大小的词表。

## 文件

| 文件 | 用途 |
|---|---|
| `bpe.py` | `BPETokenizer`：训练、编码、解码、保存与加载 |
| `corpus.py` | 流式读取本地 Parquet，统计文档数 |
| `train.py` | 在完整训练集上训练词表 |
| `inference.py` | 加载词表，展示原文、token ID、解码结果和还原检查 |
| `run.sh` | 依次执行完整训练和编码解码演示 |
| `evaluate.py` | 在验证集上检查还原效果并统计 token 数 |
| `tinystories-bpe-16000/` | 已训练词表、合并规则与训练记录 |

数据位于 `/user-data/chaojh26/TinyStories/`。
训练使用全部四个 train 分片，共 2,119,719 篇故事、1,902,941,607 个 UTF-8 字节。
按文件和行顺序逐批读取，不抽样，不使用随机种子；验证集独立保留。

## 使用

在项目根目录 `/root/chaojh26/Advanced-Machine-Learning` 执行：

```bash
# 一次执行完整训练和推理演示，具体参数写在 run.sh 中
bash tokenizer/run.sh

# 只用当前已训练好的词表运行演示
python tokenizer/inference.py --text "Hello, 世界！🙂"

# 手动分步执行训练和推理
python tokenizer/train.py --output-dir tokenizer/tinystories-new
python tokenizer/inference.py --model-dir tokenizer/tinystories-new

# 在完整验证集上评估
python tokenizer/evaluate.py --max-documents 21990
```

`run.sh` 使用当前环境中的 `python`，显式列出训练和推理的全部参数。
需要更换输出目录时，同步修改脚本中的 `--output-dir` 和 `--model-dir`；
相对目录以 `tokenizer/` 为基准。
训练入口会拒绝覆盖已有词表。这里的推理指文本编码与解码。

```python
from tokenizer import BPETokenizer

model = BPETokenizer.load("tokenizer/tinystories-bpe-16000")
text = "Hello, 世界！🙂"
ids = model.encode(text)
print(ids)
assert model.decode(ids) == text
```

## 词表 ID

初始词表覆盖全部 256 种字节，ID 为 `0..255`，ID 与字节值相同。
学习到的合并 token 从 ID `256` 开始。普通 UTF-8 文本均可通过字节表示。

## 实现与结果

使用 GPT-2 风格正则预切分，保留 Unicode 和空白。按片段出现次数加权统计相邻 pair，
每轮合并频次最高的一对；频次相同按左右 token ID 排序。
倒排索引与堆用于增量更新统计。每十万文档、每五百条合并规则记录进度。

目标词表包含 256 个基础字节和 15,744 个学习符号。
完整训练统计保存在 `tokenizer.json` 中，验证统计保存在 `validation_metrics.json` 中。
评估报告中的 `bytes_per_token` 是平均字节数，不是语言模型 BPB。

运行 `train.py` 生成 `tokenizer.json`、`vocab.json`、`merges.json` 和 `training_log.jsonl`。
运行 `evaluate.py` 生成 `validation_metrics.json` 和 `examples.json`。


