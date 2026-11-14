# Scaling Law 实验配置

使用 8 个小模型，每个从随机初始化开始训练 TinyStories 的 1 个 epoch。
已完成的 234M 主模型作为外推校验，不参与拟合。

| 名称 | 层数 | 隐藏维度 | 注意力头数 | FFN 维度 | 总参数量 |
|---|---:|---:|---:|---:|---:|
| s01 | 4 | 128 | 2 | 512 | 4.88M |
| s02 | 4 | 192 | 3 | 768 | 7.92M |
| s03 | 6 | 256 | 4 | 1024 | 12.91M |
| s04 | 6 | 320 | 5 | 1280 | 17.62M |
| s05 | 8 | 384 | 6 | 1536 | 26.45M |
| s06 | 8 | 512 | 8 | 2048 | 41.56M |
| s07 | 12 | 640 | 10 | 2560 | 79.48M |
| s08 | 12 | 768 | 12 | 3072 | 109.53M |

配置保存在 `scaling_models.csv`。每头维度均为 64，FFN 为隐藏维度的 4 倍。
词表、上下文、分词器、打包方式、数据顺序和优化配方保持一致。
每组使用两卡、每卡 batch 16、梯度累积 4、seed 42，处理相同的训练块。
当前数据下各模型均训练 3,861 steps，累计有效训练 token 数相同。

训练前、每 100 步和最后一步都计算完整验证集的 loss、PPL、BPB。
每组记录 `metrics.jsonl` 中的 `tokens_seen` 和 `estimated_flops=6ND`。
缩小模型不改变训练步数，但每步计算量会下降；实际耗时还受 GPU 利用率、验证和保存开销影响。
这些实验关闭文本采样以节省时间，模型训练及验证不受影响。

在项目根目录启动整个队列：

```bash
nohup setsid bash pretrain/run_scaling.sh > pretrain/scaling.log 2>&1 < /dev/null &
tail -f pretrain/scaling.log
```

脚本顺序训练八个模型。每组有独立目录与 W&B run，结果位于：

```text
/user-data/chaojh26/pretrain/scaling/tinystories-1/s01/
...
/user-data/chaojh26/pretrain/scaling/tinystories-1/s08/
```

每组完整控制台输出在该目录的 `train.log`，例如：

```bash
tail -f /user-data/chaojh26/pretrain/scaling/tinystories-1/s01/train.log
```

队列使用 `--wandb-mode online`，训练指标实时上传 W&B，同时保存本地记录。
在线访问失败时停止队列，不自动切换到离线模式。已完成的 s01～s04 在线实验保留原记录。

脚本跳过已生成 `final_metrics.json` 的模型，遇到失败会停止队列。
中断但未完成的模型保留现有训练入口的覆盖保护，需要单独使用原配置加 `--resume` 恢复，
或为新实验选择新的输出目录。

## 后续分析口径

固定数据量的模型尺寸拟合使用训练终点的验证结果：
读取 `metrics.jsonl` 中 `step=3861`、`split=validation` 且未标记 `final` 的记录。
`final_metrics.json` 对应最佳 checkpoint，最佳模型可能早于训练终点，二者的训练预算可能不同。

经验计算包络使用不同模型训练途中记录的 `(estimated_flops, val loss)`：
在相同计算预算下比较，只在各模型实际覆盖的预算范围内插值，选择最低 loss。
初始 step 0 不用于对数坐标插值；训练结束后重新评估最佳模型的 `final` 记录不作为新计算预算点。
该包络依赖当前统一训练配方；长训练中途的学习率与专门短训练的学习率周期不同，需在报告中说明。

所有拟合统一使用实际总参数量；s01、s02 的 Embedding 与输出头占比较大，
可另外报告排除最小模型后的拟合敏感性，避免把小规模偏离当作普遍规律。
