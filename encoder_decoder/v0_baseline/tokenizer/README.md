# Tokenizer

## 说明

基于 `bert-base-multilingual-cased` 底座重新训练的 BPE tokenizer，用于本项目中英翻译任务的编码和解码。

## 配置

| 项目 | 值 |
|------|-----|
| 底座 tokenizer | bert-base-multilingual-cased |
| 类型 | BPE（Fast tokenizer，Rust 实现） |
| vocab_size | 16000 |
| 训练语料 | 与训练集相同的中英平行语料 |
| 是否 src/tgt 共用 | 是 |

## 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| PAD | 0 | 填充，loss 计算中忽略 |
| UNK | 1 | 未知词 |
| BOS (CLS) | 2 | 句子起始 |
| EOS (SEP) | 3 | 句子结束 |
| MASK | 4 | 掩码 |

## 文件

| 文件 | 说明 |
|------|------|
| `tokenizer.json` | Fast tokenizer 核心文件，包含词表、分词规则、normalizer 等 |
| `tokenizer_config.json` | tokenizer 配置（特殊 token 定义等） |

## 使用方式

训练脚本在启动时会自动从此目录加载底座 tokenizer，并使用训练语料重新训练为 16000 词表的专属 tokenizer。

```python
from transformers import AutoTokenizer
base_tokenizer = AutoTokenizer.from_pretrained("tokenizer/", use_fast=True)
tokenizer = base_tokenizer.train_new_from_iterator(texts, vocab_size=16000)
```

训练结束后，新 tokenizer 会保存到训练输出目录（`minimal_transformer_en_zh_opus_outputs/tokenizer/`），供推理脚本使用。
