# Data

本目录保存 Encoder-Decoder Transformer v0_baseline 使用的中英翻译数据。

## Source

数据来自公开数据集 [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) 的 en-zh 子集。

## Format

完整数据集使用 HuggingFace Arrow 格式保存，每条记录结构为：

```json
{"translation": {"en": "...", "zh": "..."}}
```

本地目录：

```text
opus100_en_zh_local/
├── train/
├── validation/
├── test/
└── dataset_dict.json
```

原始 split 规模：

| split | records |
|-------|---------|
| train | 1000000 |
| validation | 2000 |
| test | 2000 |

训练脚本当前使用 train 前 500000 条 raw records 和 validation 前 2000 条 raw records。`TranslationDataset` 会在运行时过滤空字符串样本和 token 数过短样本。

## Sample Data

`sample_data.jsonl` 包含 200 条真实样例，每行一个 JSON：

```json
{"en": "...", "zh": "...", "source": "opus100_en_zh"}
```

该文件直接进入普通 Git，方便快速查看数据格式。

## Git LFS

完整 Arrow 数据文件使用 Git LFS 管理。`sample_data.jsonl` 不使用 Git LFS。
