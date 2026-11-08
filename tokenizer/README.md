# BPE 分词器

自行实现的字节级 BPE，使用完整 TinyStories 训练集学习 16,000 大小的词表。

## 文件

| 文件 | 用途 |
|---|---|
| `bpe.py` | `BPETokenizer`：训练、编码、解码、保存与加载 |
| `corpus.py` | 流式读取本地 Parquet，统计文档数 |
| `train.py` | 在完整训练集上训练词表 |
| `infer.py` | 使用自训练或参考分词器编码解码文本 |
| `run.sh` | 依次执行完整训练和编码解码演示 |
| `evaluate.py` | 按数据集和 split 统计完整语料的 token 数与还原情况 |
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
python tokenizer/infer.py --text "Hello, 世界！🙂"

# 手动分步执行训练和推理
python tokenizer/train.py --output-dir tokenizer/tinystories-new
python tokenizer/infer.py --model-dir tokenizer/tinystories-new

# 使用自训练词表评估完整 TinyStories 训练集
python tokenizer/evaluate.py \
    --model-dir tokenizer/tinystories-bpe-16000 \
    --dataset tinystories --split train \
    --output-dir tokenizer/results/tinystories-custom-train
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
完整训练统计保存在 `tokenizer.json` 中，新的评估结果写入指定的 `--output-dir`。
评估报告中的 `bytes_per_token` 是平均字节数，不是语言模型 BPB。

运行 `train.py` 生成 `tokenizer.json`、`vocab.json`、`merges.json` 和 `training_log.jsonl`。
运行 `evaluate.py` 在 `--output-dir` 中生成 `metrics.json` 和 `examples.json`。



## 参考分词器与数据集评估

`--model-dir tinystories-bpe-16000` 从本项目训练目录加载 `BPETokenizer`，
也可直接指定其他自训练词表的目录路径。
`--model-dir flexitok/bpe_fw_edu_16000` 直接调用
`Tokenizer.from_pretrained("flexitok/bpe_fw_edu_16000")`。
两种加载方式直接写在 `evaluate.py` 和 `infer.py` 中。

```bash
# 参考分词器推理
python tokenizer/infer.py \
    --model-dir flexitok/bpe_fw_edu_16000 \
    --text "Hello, 世界！🙂"

# 参考分词器评估完整 TinyStories 验证集
python tokenizer/evaluate.py \
    --model-dir flexitok/bpe_fw_edu_16000 \
    --dataset tinystories --split validation \
    --output-dir tokenizer/results/tinystories-reference-validation

# 参考分词器评估 FineWeb 验证集
python tokenizer/evaluate.py \
    --model-dir flexitok/bpe_fw_edu_16000 \
    --dataset fineweb --split validation --seed 42 \
    --output-dir tokenizer/results/fineweb-reference-validation

# 自训练词表评估 FineWeb 训练集
python tokenizer/evaluate.py \
    --model-dir tokenizer/tinystories-bpe-16000 \
    --dataset fineweb --split train --seed 42 \
    --output-dir tokenizer/results/fineweb-custom-train
```

`--split` 默认 `train`，也可选择 `validation`，始终处理完整的所选 split。
`--output-dir` 必须指定。不同分词器、数据集和 split 建议使用不同输出目录。
`--data-dir` 可自行指定；默认 TinyStories 路径为 `/user-data/chaojh26/TinyStories/`，
FineWeb 路径为 `/user-data/chaojh26/FineWeb-Edu/sample-10BT/`。

TinyStories 沿用官方训练集和验证集。FineWeb 按固定文件顺序给文档编号，
通过 `random.Random(seed).sample()` 无放回随机抽取 `总文档数 // 100` 篇作为验证集，
其余文档作为训练集。`--seed` 默认 42，仅影响 FineWeb 的划分。
相同数据和 seed 得到相同划分；更换 seed 会改变入选文档。
两个 split 互斥且覆盖全部文档，仍按原文件顺序流式读取，不复制原始文件。
后续读取 FineWeb 时可复用
`corpus.iter_documents(data_dir, split, dataset="fineweb", seed=42)`。
本机 9,672,101 篇 FineWeb 文档分为 9,575,380 篇训练文档和 96,721 篇验证文档。
99:1 按文档数划分，token 数的比例由文本长度决定。

`metrics.json` 记录分词器来源、词表大小、数据目录、文件列表、split、随机划分规则与 seed、
文档数、全部 token 出现次数、UTF-8 字节数、字符数、平均 bytes/token、原文还原匹配数、
归一化后匹配数和评估耗时。重复 token 全部计入，不填充、不截断、不额外添加边界 token。
`examples.json` 保存前三篇文档的 `text`、`ids`、`decoded` 和 `is_match`。

参考分词器执行 NFC 归一化。解码结果与原文不同时，评估记录差异并继续；
`normalized_matched_documents` 单独统计解码结果与归一化后原文是否一致。
`infer.py` 也会如实输出 `is_match`，归一化引起的差异不会作为运行错误退出。
