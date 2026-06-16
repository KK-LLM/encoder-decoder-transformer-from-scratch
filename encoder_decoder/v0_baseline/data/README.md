# Data

## 数据来源

[Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) 的 en-zh（英文→中文）子集。opus-100 是 OPUS 项目收集的多语言平行语料，许可证为 CC-BY-4.0。

## 数据格式

HuggingFace Arrow 格式（Translation 类型），每条记录结构：

```json
{"translation": {"en": "英文句子", "zh": "中文翻译"}}
```

## 数据规模

| 划分 | 原始样本数 | 本次使用 |
|------|----------|---------|
| train | 1,000,000 | 前 500,000（有效约 499,989） |
| validation | 2,000 | 2,000 |
| test | 2,000 | 未使用 |

## 本目录文件

| 文件 | 说明 |
|------|------|
| `sample_data.jsonl` | 200 条数据样例，方便快速查看数据格式 |
| `opus100_en_zh_local/` | 完整 Arrow 数据集（Git LFS），用于实际训练 |

## 获取完整数据

完整数据集可以从 HuggingFace Hub 直接下载：

```python
from datasets import load_dataset
ds = load_dataset("Helsinki-NLP/opus-100", "en-zh")
```

或在项目中加载本地副本（需先 clone 并安装 Git LFS）：

```python
from datasets import Dataset
train_ds = Dataset.load_from_disk("data/opus100_en_zh_local/train")
```
