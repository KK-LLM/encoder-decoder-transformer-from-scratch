# Encoder-Decoder Transformer from Scratch

> Encoder-Decoder Transformer from scratch learning and training lab.

这个仓库用于整理一个手写 Encoder-Decoder Transformer 英文到中文翻译项目。当前版本是 `v0_baseline`，重点展示从零实现模型结构、训练闭环、推理闭环、tokenizer 使用和 checkpoint 保存机制。v0_baseline 训练已完成，已提交 baseline checkpoint、训练日志、推理日志、固定测试集分析和错误分析。

## Project Overview

本项目聚焦原始 Transformer Encoder-Decoder 架构，不依赖 `nn.Transformer` 封装。训练脚本内嵌模型源码和训练流程，推理脚本内嵌同结构模型源码和 greedy decode 流程，方便面试官直接阅读完整 pipeline。

当前仓库只包含 Encoder-Decoder Transformer 相关内容。Decoder-only from scratch 后续会单独新建仓库，不属于当前仓库。

## Why This Repo

- 手写 Transformer 基础模块：`TokenEmbedding`、`PositionalEncoding`、`LayerNorm`、`MultiHeadAttention`、`Encoder`、`Decoder`、`Generator`
- 搭建训练闭环：数据加载、tokenizer 训练、mask 构造、loss 计算、Noam learning rate、checkpoint 保存
- 搭建推理闭环：tokenizer 加载、checkpoint 加载、`src_mask` / `tgt_mask` 构造、greedy decode
- 保留清晰的版本目录和文档说明，便于后续按真实实验进展持续更新

## Current Progress

| 项目 | 版本 | 状态 |
|------|------|------|
| Encoder-Decoder Transformer | [v0_baseline](./encoder_decoder/v0_baseline/) | 训练已完成，已提交 checkpoint 和完整日志 |

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── requirements.txt
├── docs/
│   ├── experiment_notes.md
│   ├── architecture_notes.md
│   └── checkpoint_and_reproducibility.md
├── encoder_decoder/
│   ├── README.md
│   └── v0_baseline/
│       ├── README.md
│       ├── src/
│       │   ├── train_encoder_decoder.py
│       │   └── infer_encoder_decoder.py
│       ├── data/
│       │   ├── README.md
│       │   ├── sample_data.jsonl
│       │   └── opus100_en_zh_local/
│       ├── tokenizer/
│       │   ├── README.md
│       │   ├── bert_base_multilingual_cased_tokenizer/
│       │   └── trained_tokenizer_16000/
│       ├── checkpoints/
│       │   ├── README.md
│       │   └── checkpoint_epoch_20.pt
│       ├── logs/
│       │   ├── README.md
│       │   ├── train_metrics.csv
│       │   ├── train_log_summary.md
│       │   ├── opus_train_log.log
│       │   ├── inference_eval_epoch20.jsonl
│       │   ├── fixed_eval_epoch20.jsonl
│       │   ├── inference_metrics_epoch20.csv
│       │   └── tokenizer_term_check_epoch20.txt
│       └── results/
│           ├── README.md
│           ├── translation_examples.md
│           ├── inference_report.md
│           └── inference_examples.md
└── scripts/
    ├── run_train_encoder_decoder.sh
    └── run_infer_encoder_decoder.sh
```

## Encoder-Decoder Baseline

`v0_baseline` 是当前 baseline 版本，用于验证英文到中文机器翻译任务上的完整实现闭环。该版本强调源码可读性和 pipeline 完整性，不追求达到最佳翻译效果。

## Model Architecture

训练脚本中的模型配置以真实代码为准：

| 配置 | 值 |
|------|----|
| architecture | Transformer Encoder-Decoder, Post-LN |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `num_heads` | 8 |
| `num_layers` | 8 encoder layers + 8 decoder layers |
| `dropout` | 0.1 |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| tokenizer vocab size | 16000, runtime-trained from the base tokenizer |

模型源码目前内嵌在训练脚本和推理脚本中，未拆分为独立 `model.py`。

## Training Pipeline

训练脚本位置：

```text
encoder_decoder/v0_baseline/src/train_encoder_decoder.py
```

脚本执行的主要流程：

1. 加载本地 OPUS-100 en-zh Arrow 数据集
2. 基于 `bert_base_multilingual_cased_tokenizer` 和训练语料重新训练 16000 vocab tokenizer
3. 构建 `TranslationDataset`，过滤空字符串样本和 token 数过短样本
4. 构建 Encoder-Decoder Transformer
5. 使用 Adam + Noam learning rate 训练
6. 计算 train / valid loss
7. 每 4 个 epoch 保存一次 checkpoint，训练结束保存 final model

## Inference Pipeline

推理脚本位置：

```text
encoder_decoder/v0_baseline/src/infer_encoder_decoder.py
```

推理脚本支持加载训练输出目录中的 tokenizer 和 checkpoint，构造 `src_mask` / `tgt_mask`，并使用 greedy decode 执行自回归翻译。v0_baseline 训练完成后已使用 epoch 20 checkpoint 执行 inference evaluation，包含 train / validation / test 抽样和 fixed_simple / fixed_terms / fixed_logic 三类专项测试，详见 [inference_report.md](./encoder_decoder/v0_baseline/results/inference_report.md)。

## Tokenizer

本版本上传的 tokenizer 资源是训练脚本使用的 base tokenizer snapshot：

```text
encoder_decoder/v0_baseline/tokenizer/bert_base_multilingual_cased_tokenizer/
```

训练脚本会基于该 tokenizer 和训练语料运行 `train_new_from_iterator(..., vocab_size=16000)`，生成源语言和目标语言共用的 16000 vocab tokenizer。训练完成后生成的 tokenizer 已随着 best checkpoint 上传至：

```text
encoder_decoder/v0_baseline/tokenizer/trained_tokenizer_16000/
```

## Dataset

数据来自公开数据集 [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) 的 en-zh 子集，本地保存为 HuggingFace Arrow 格式：

```text
encoder_decoder/v0_baseline/data/opus100_en_zh_local/
```

完整数据集通过 Git LFS 管理。`sample_data.jsonl` 包含 200 条真实中英样例，方便快速查看数据格式。

## Checkpoint Support

训练脚本支持 checkpoint 保存：

- 每 4 个 epoch 保存一次：`checkpoint_epoch_{epoch}.pt`
- 训练结束保存：`final_model.pt`
- 每次保存 checkpoint 时同步保存训练时生成的 tokenizer

当前脚本只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，因此当前版本不支持自动续训。v0_baseline 训练完成后已提交 best checkpoint：

```text
encoder_decoder/v0_baseline/checkpoints/checkpoint_epoch_20.pt
```

该 checkpoint 基于 valid_loss 全局最低（epoch 20，valid_loss=3.4693）选择，使用 Git LFS 管理。

## Current Training Status

v0_baseline 训练已完成（48 epoch）。valid_loss 在 epoch 20 达到最低（3.4693），此后出现 overfitting 趋势。选择 checkpoint_epoch_20.pt 作为 baseline checkpoint。完整训练指标见 [train_metrics.csv](./encoder_decoder/v0_baseline/logs/train_metrics.csv)，训练摘要见 [train_log_summary.md](./encoder_decoder/v0_baseline/logs/train_log_summary.md)。

## Limitations

1. 当前版本是 `v0_baseline`，主要用于展示从零实现 Encoder-Decoder Transformer 的完整闭环。
2. v0_baseline 训练已完成，bug 已被提交 checkpoint、训练日志、推理日志和固定测试集分析。
3. 当前不提供 BLEU / chrF 等自动评测指标，推理使用 greedy decode。
4. 推理脚本当前使用 greedy decode。
5. 模型源码内嵌在 train / infer 脚本中，未拆分为独立 `model.py`。
6. checkpoint 当前只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，也不支持自动续训。
7. 当前版本更强调完整 pipeline 和源码可读性，而不是最终翻译效果最优。
8. 在 OPUS-100 UN 语体内模型可生成流畅中文，但技术术语和日常英语泛化能力受限，详见 [inference_report](./encoder_decoder/v0_baseline/results/inference_report.md)。

## How to Run

安装依赖：

```bash
pip install -r requirements.txt
```

运行训练：

```bash
bash scripts/run_train_encoder_decoder.sh
```

训练产生 checkpoint 和 tokenizer 后，可运行推理：

```bash
bash scripts/run_infer_encoder_decoder.sh
```

## Notes for Reviewers

推荐阅读路径：

1. [v0_baseline README](./encoder_decoder/v0_baseline/README.md)
2. [训练脚本](./encoder_decoder/v0_baseline/src/train_encoder_decoder.py)
3. [推理脚本](./encoder_decoder/v0_baseline/src/infer_encoder_decoder.py)
4. [训练摘要](./encoder_decoder/v0_baseline/logs/train_log_summary.md)
5. [推理报告](./encoder_decoder/v0_baseline/results/inference_report.md)
6. [架构说明](./docs/architecture_notes.md)
7. [checkpoint 与复现说明](./docs/checkpoint_and_reproducibility.md)

后续 Encoder-Decoder 更新会根据实际实验进展补充到对应版本目录中，避免在当前 baseline 阶段提前承诺过多未完成内容。
