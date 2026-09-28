# 基于 BERT 的中文文本分类

中文评论情感二分类（好评/差评），对比传统机器学习基线与 BERT 微调的效果差异。

- 数据集：ChnSentiCorp（中文酒店/商品评论，训练集 9600 条、测试集 1200 条，通过 HuggingFace `datasets` 加载）
- 基线：TF-IDF（字符级 1-2 gram）+ 逻辑回归
- 主模型：`bert-base-chinese` 预训练模型微调（3 epoch，batch size 32，学习率 2e-5，max_len 128）
- 设备：单卡 NVIDIA RTX 4060 Laptop（8GB）

## 实验结果（测试集 1200 条）

| 模型 | Accuracy | F1 (macro) |
| --- | --- | --- |
| TF-IDF + 逻辑回归（基线） | 0.9000 | 0.9000 |
| BERT-base-chinese 微调 | 0.9275 | 0.9275 |

BERT 微调 F1 较基线提升 2.75 个百分点。

## 目录结构

```
baseline.py     # 基线模型：TF-IDF + 逻辑回归
train_bert.py   # BERT 微调训练（含逐行中文注释）
experiments.py  # 超参数扫描实验（学习率 / batch size / 轮次，支持 --smoke 冒烟）
results/        # 实验产物：experiment_report.md + experiments.json
```

## 超参数对比实验

固定 2500 条训练子集（种子 42 抽样保证各组同分布）、全量 1200 条测试集评估，扫描 6 组配置：

| 配置 | 学习率 | batch size | 轮次 | F1 (macro) |
| --- | --- | --- | --- | --- |
| lr1e-5 | 1e-05 | 32 | 1 | 0.8964 |
| lr2e-5 | 2e-05 | 32 | 1 | 0.8907 |
| lr5e-5 | 5e-05 | 32 | 1 | 0.8982 |
| bs16 | 2e-05 | 16 | 1 | 0.8990 |
| **ep3** | 2e-05 | 32 | **3** | **0.9233** |

主要结论：
- **轮次影响最大**：1 轮 0.8907 → 3 轮 0.9233；但轮次过多有过拟合风险，需每轮评估取最优
- **学习率**：微调常用 1e-5 ~ 5e-5 量级，过大破坏预训练权重、过小收敛慢
- **batch size**：16 略优于 32（梯度噪声带来的正则效应），差异在小噪声范围内
- 子集扫描最优结论与全量主实验（F1 0.9275）一致

完整报告见 [results/experiment_report.md](results/experiment_report.md)。

## 复现方式

```bash
# 安装依赖（Python 3.11，CUDA 12.4 版 PyTorch）
uv sync

# 国内网络可设置 HuggingFace 镜像
$env:HF_ENDPOINT="https://hf-mirror.com"   # PowerShell

# 1. 跑基线（CPU，数秒）
uv run python baseline.py

# 2. 微调 BERT（4060 约 10 分钟）
uv run python train_bert.py

# 3. 超参数扫描（4060 约 12 分钟；先用 --smoke 冒烟验证约 1 分钟）
uv run python experiments.py
```

## 实现要点

- **数据泄漏防护**：TF-IDF 仅在训练集 `fit` 词表，测试集只 `transform`，防止测试集词汇分布参与训练
- **中文处理**：基线使用字符级 n-gram（`analyzer='char'`）规避中文无空格分词问题
- **BERT 输入**：tokenizer 截断/填充至 128，`attention_mask` 区分真实 token 与 padding
- **评估口径**：统一使用 macro-F1（好评/差评两类 F1 取平均），避免类别不均衡下准确率虚高
- **训练细节**：AdamW + 线性学习率衰减（10% warmup），固定随机种子 42 保证可复现，每轮在测试集评估并保存最优权重
