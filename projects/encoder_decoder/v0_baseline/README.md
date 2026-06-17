# Encoder-Decoder Transformer v0_baseline

`v0_baseline` 是手写 Transformer Encoder-Decoder 的 baseline 版本，任务是英文到中文单向机器翻译。该版本重点展示模型源码、训练闭环、推理闭环、tokenizer 训练与 checkpoint 保存机制。v0_baseline 训练已完成，已提交 checkpoint、训练日志、推理日志、固定测试集分析和错误分析。

## Task

| 项目 | 说明 |
|------|------|
| task | English-to-Chinese machine translation |
| source language | `en` |
| target language | `zh` |
| dataset | Helsinki-NLP/opus-100 en-zh |

## Source Files

| 文件 | 说明 |
|------|------|
| [src/train_encoder_decoder.py](./src/train_encoder_decoder.py) | 手写模型源码 + 训练闭环 |
| `src/infer_encoder_decoder.py` | 手写模型源码 + 推理闭环 |
| [data/sample_data.jsonl](./data/sample_data.jsonl) | 200 条真实中英样例 |
| [tokenizer/README.md](./tokenizer/README.md) | tokenizer 使用说明 |
| [checkpoints/README.md](./checkpoints/README.md) | checkpoint 保存与当前限制 |
| [logs/](./logs/) | 训练与推理量化数据 |
| [results/](./results/) | 训练与推理分析报告 |

## Model Configuration

以下配置来自训练脚本真实参数：

| 参数 | 值 |
|------|----|
| architecture | Transformer Encoder-Decoder, Post-LN |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `num_heads` | 8 |
| `num_layers` | 8 |
| `dropout` | 0.1 |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| tokenizer vocab size | 16000 |

关键模块包括 `TokenEmbedding`、`PositionalEncoding`、`LayerNorm`、`MultiHeadAttention`、`PositionwiseFeedForward`、`Encoder`、`Decoder` 和 `Generator`。

## Training Configuration

以下配置来自训练脚本真实参数：

| 参数 | 值 |
|------|----|
| raw train range | first 500000 train records |
| raw valid range | first 2000 validation records |
| batch size | 280 |
| epochs | 48 |
| optimizer | Adam, `lr=1.0`, `betas=(0.9, 0.98)`, `eps=1e-9` |
| learning rate | Noam formula with `d_model=512` |
| scheduler | `torch.optim.lr_scheduler.LambdaLR` |
| warmup steps | 3000 |
| loss | `F.cross_entropy`, `ignore_index=PAD_ID` |
| label smoothing | 0.1 |
| gradient clipping | `max_norm=1.0` |
| mixed precision | BF16 autocast when CUDA is available |
| device selection | CUDA if available, otherwise CPU |

`TranslationDataset` 会过滤空字符串样本，并过滤 BOS + 正文 + EOS 后 token 数小于 3 的样本。过滤后的有效样本数由训练脚本运行时打印。

## Tokenizer

训练脚本使用本版本目录下的 base tokenizer snapshot：

```text
tokenizer/bert_base_multilingual_cased_tokenizer/
```

脚本会基于该 tokenizer 和训练语料重新训练一个 16000 vocab tokenizer。源语言和目标语言共用 tokenizer。`PAD_ID`、`BOS_ID`、`EOS_ID` 在脚本中从 tokenizer 动态读取，其中 BOS 使用 `[CLS]`，EOS 使用 `[SEP]`。

训练完成后生成的 tokenizer 已上传至：

```text
tokenizer/trained_tokenizer_16000/
```

该 tokenizer 与 `checkpoints/checkpoint_epoch_20.pt` 配套使用，不可混用 base tokenizer 直接推理。

## Dataset

完整数据集位于：

```text
data/opus100_en_zh_local/
```

该目录为 HuggingFace Arrow 格式，记录结构为：

```json
{"translation": {"en": "...", "zh": "..."}}
```

完整数据集使用 Git LFS 管理。`sample_data.jsonl` 是从真实 train split 中抽取的 200 条样例，普通 Git 提交。

## Checkpoints

训练脚本保存：

- `checkpoint_epoch_{epoch}.pt`，每 4 个 epoch 保存一次
- `final_model.pt`，训练结束保存
- `tokenizer/`，随 checkpoint 输出保存

v0_baseline 训练完成后已提交 best checkpoint：

```text
checkpoints/checkpoint_epoch_20.pt
```

选择依据：valid_loss 在 epoch 20 达到全局最低（3.4693）。

该 checkpoint 文件（338MB）使用 Git LFS 管理，下载前需安装 `git-lfs`。

当前 checkpoint 只包含：

```python
{
    "epoch": epoch,
    "model_state_dict": cpu_state_dict,
}
```

当前脚本不保存 optimizer / scheduler 状态，不支持自动续训。

## Inference

推理脚本支持加载训练输出目录中的 tokenizer 和 checkpoint，构造 `src_mask` / `tgt_mask`，并执行 greedy decode。v0_baseline 训练完成后使用 `checkpoint_epoch_20.pt` + `trained_tokenizer_16000/` 执行了完整推理评估，包含：

- train / validation / test 抽样推理（250 条）
- fixed_simple / fixed_terms / fixed_logic 固定测试（55 条）
- 人工复核后的错误类型分析

详见 [results/inference_report.md](./results/inference_report.md) 和 [results/inference_examples.md](./results/inference_examples.md)。

## Current Training Status

v0_baseline 训练已完成（48 epoch）。valid_loss 在 epoch 20 达到最低（3.4693），此后出现 overfitting 趋势。完整指标见 [logs/train_metrics.csv](./logs/train_metrics.csv)，训练摘要见 [logs/train_log_summary.md](./logs/train_log_summary.md)。

## Limitations

1. 当前版本是 `v0_baseline`，主要用于展示完整 Encoder-Decoder Transformer pipeline。
2. 当前不提供 BLEU / chrF 等自动评测指标，推理使用 greedy decode。
3. 推理脚本当前使用 greedy decode。
4. 模型源码内嵌在 train / infer 脚本中，未拆分为独立 `model.py`。
5. checkpoint 当前只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，不支持自动续训。
6. 当前版本更强调完整 pipeline 和源码可读性，而不是最终翻译效果最优。
7. 在 OPUS-100 UN 语体内模型可生成流畅中文，但技术术语和日常英语泛化能力受限。

## Run

从仓库根目录运行训练：

```bash
bash scripts/run_train_encoder_decoder.sh
```

训练产生 checkpoint 后运行推理：

```bash
bash scripts/run_infer_encoder_decoder.sh
```
