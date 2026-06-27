# v1_curriculum Tokenizer

该目录保存 `v1_curriculum` 使用的统一 tokenizer。

## Tokenizer Directory

```text
en_zh_stage_tokenizer_48000/
```

该 tokenizer 基于 Stage1 到 Stage5 的课程学习数据构造，词表大小为 48,000。Stage1 到 Stage5 使用同一个 tokenizer 和同一个模型结构，避免不同阶段之间因为 tokenizer 变化导致 checkpoint 无法连续传递。

## Files

```text
en_zh_stage_tokenizer_48000/
├── curriculum_tokenizer_metadata.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt
```

## Special Tokens

训练脚本会校验以下特殊 token id：

| token | 用途 | expected id |
|------|------|-------------:|
| `[PAD]` | padding token | 0 |
| `[UNK]` | unknown token | 1 |
| `[CLS]` | BOS token | 2 |
| `[SEP]` | EOS token | 3 |
| `[MASK]` | mask token | 4 |

## Runtime Behavior

v1 训练脚本不会在训练时重新训练 tokenizer，而是直接加载该预构建 tokenizer：

```text
tokenizer/en_zh_stage_tokenizer_48000/
```

训练脚本会从 tokenizer 动态读取：

- `PAD_ID`
- `BOS_ID`
- `EOS_ID`

并检查词表大小是否为 48,000。

## Difference from v0_baseline

`v0_baseline` 使用的是基于 OPUS-100 en-zh 训练语料重新训练的 16,000 vocab tokenizer。

`v1_curriculum` 使用的是基于 Stage1 到 Stage5 课程数据构造的 48,000 vocab tokenizer。

这两个 tokenizer 不应混用：

- v0 checkpoint 应使用 v0 tokenizer
- v1 checkpoint 应使用 v1 tokenizer

## Purpose

v1 tokenizer 的设计目标是提升以下内容的切分稳定性：

- 基础日常英语
- 普通书面表达
- 逻辑连接结构
- Transformer / tokenizer / encoder / decoder / checkpoint 等技术术语
- mixed regression 和 anti-pollution 样本中的技术语境词

更大的词表不是单独追求规模，而是配合 Stage1 到 Stage5 的课程数据，让模型在基础句、技术句和复杂逻辑句之间保持更稳定的 token 表示。
