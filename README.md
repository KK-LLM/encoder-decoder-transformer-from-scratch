# Encoder-Decoder Transformer from Scratch

> Encoder-Decoder Transformer from scratch learning and training lab.

这个仓库用于整理一个手写 Encoder-Decoder Transformer 英文到中文翻译项目。当前版本是 `v0_baseline`，重点展示从零实现模型结构、训练闭环、推理闭环、tokenizer 使用和 checkpoint 保存机制。当前训练仍在进行中，本轮不提交训练日志、推理日志、推理样例或模型权重。

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
| Encoder-Decoder Transformer | [v0_baseline](./projects/encoder_decoder/v0_baseline/) | 训练进行中 |

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
├── projects/
│   └── encoder_decoder/
│       ├── README.md
│       └── v0_baseline/
│           ├── README.md
│           ├── src/
│           │   ├── train_encoder_decoder.py
│           │   └── infer_encoder_decoder.py
│           ├── data/
│           │   ├── README.md
│           │   ├── sample_data.jsonl
│           │   └── opus100_en_zh_local/
│           ├── tokenizer/
│           │   ├── README.md
│           │   └── bert_base_multilingual_cased_tokenizer/
│           └── checkpoints/
│               └── README.md
└── scripts/
    ├── run_train_encoder_decoder.sh
    └── run_infer_encoder_decoder.sh
```

## Encoder-Decoder Baseline

`v0_baseline` 是当前 baseline 版本，用于验证英文到中文机器翻译任务上的完整实现闭环。该版本强调源码可读性和 pipeline 完整性，不宣称达到最佳翻译效果。

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
projects/encoder_decoder/v0_baseline/src/train_encoder_decoder.py
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
projects/encoder_decoder/v0_baseline/src/infer_encoder_decoder.py
```

推理脚本支持加载训练输出目录中的 tokenizer 和 checkpoint，构造 `src_mask` / `tgt_mask`，并使用 greedy decode 执行自回归翻译。由于当前训练仍在进行中，本轮不提交推理样例和推理效果结论。

## Tokenizer

本版本上传的 tokenizer 资源是训练脚本使用的 base tokenizer snapshot：

```text
projects/encoder_decoder/v0_baseline/tokenizer/bert_base_multilingual_cased_tokenizer/
```

训练脚本会基于该 tokenizer 和训练语料运行 `train_new_from_iterator(..., vocab_size=16000)`，生成源语言和目标语言共用的 16000 vocab tokenizer。训练完成后，生成的 tokenizer 会随 checkpoint 输出到训练输出目录。

## Dataset

数据来自公开数据集 [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) 的 en-zh 子集，本地保存为 HuggingFace Arrow 格式：

```text
projects/encoder_decoder/v0_baseline/data/opus100_en_zh_local/
```

完整数据集通过 Git LFS 管理。`sample_data.jsonl` 包含 200 条真实中英样例，方便快速查看数据格式。

## Checkpoint Support

训练脚本支持 checkpoint 保存：

- 每 4 个 epoch 保存一次：`checkpoint_epoch_{epoch}.pt`
- 训练结束保存：`final_model.pt`
- 每次保存 checkpoint 时同步保存训练时生成的 tokenizer

当前脚本只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，因此当前版本不支持自动续训。本轮不提交任何真实权重文件。

## Current Training Status

当前训练仍在进行中。本轮 GitHub 提交只整理项目结构、代码、数据说明、tokenizer 说明、checkpoint 说明和当前版本限制，不写最终训练结果、不写最佳 epoch、不写 best valid loss、不写推理效果结论。

## Limitations

1. 当前版本是 `v0_baseline`，主要用于展示从零实现 Encoder-Decoder Transformer 的完整闭环。
2. 当前训练仍在进行中，本轮不提交训练日志、推理日志、推理样例和 checkpoint。
3. 当前不提供最终训练指标、BLEU / chrF 或翻译质量结论。
4. 推理脚本当前使用 greedy decode。
5. 模型源码内嵌在 train / infer 脚本中，未拆分为独立 `model.py`。
6. checkpoint 当前只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，也不支持自动续训。
7. 当前版本更强调完整 pipeline 和源码可读性，而不是最终翻译效果最优。

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

1. [v0_baseline README](./projects/encoder_decoder/v0_baseline/README.md)
2. [训练脚本](./projects/encoder_decoder/v0_baseline/src/train_encoder_decoder.py)
3. [推理脚本](./projects/encoder_decoder/v0_baseline/src/infer_encoder_decoder.py)
4. [架构说明](./docs/architecture_notes.md)
5. [checkpoint 与复现说明](./docs/checkpoint_and_reproducibility.md)

后续 Encoder-Decoder 更新会根据实际实验进展补充到对应版本目录中，避免在当前 baseline 阶段提前承诺过多未完成内容。
