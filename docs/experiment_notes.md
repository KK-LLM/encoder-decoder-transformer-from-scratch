# Experiment Notes

本文记录当前 `v0_baseline` 的实验设计和训练完成后的结果摘要。

## Training Result Summary

| 指标 | 值 |
|---|---|
| epochs | 48 |
| best valid_loss | 3.4693（epoch 20） |
| train_loss at epoch 20 | 3.2443 |
| overfitting | epoch 20 后 valid_loss 波动上升 |
| best checkpoint | `checkpoint_epoch_20.pt` |

完整指标见 [logs/train_metrics.csv](../projects/encoder_decoder/v0_baseline/logs/train_metrics.csv)，训练摘要见 [logs/train_log_summary.md](../projects/encoder_decoder/v0_baseline/logs/train_log_summary.md)。推理结果见 [results/inference_report.md](../projects/encoder_decoder/v0_baseline/results/inference_report.md)。

## Objective

本轮实验用于端到端验证手写 Encoder-Decoder Transformer 的训练、验证、推理完整闭环，确保 pipeline 无阻塞性缺陷；同时在公开 OPUS-100 en-zh 数据集上初步运行多 epoch 训练。

## Configuration

以下配置以训练脚本真实参数为准：

| 参数 | 值 |
|------|----|
| task | English-to-Chinese translation |
| source language | `en` |
| target language | `zh` |
| dataset | Helsinki-NLP/opus-100 en-zh |
| raw train range | first 500000 train records |
| raw validation range | first 2000 validation records |
| model | Transformer Encoder-Decoder, Post-LN |
| `d_model` | 512 |
| `d_ff` | 2048 |
| `num_heads` | 8 |
| `num_layers` | 8 |
| `dropout` | 0.1 |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| tokenizer vocab size | 16000 |
| batch size | 280 |
| epochs | 48 |
| optimizer | Adam, `lr=1.0`, `betas=(0.9, 0.98)`, `eps=1e-9` |
| scheduler | LambdaLR + Noam formula |
| warmup steps | 3000 |
| loss | cross entropy, `ignore_index=PAD_ID` |
| label smoothing | 0.1 |
| gradient clipping | `max_norm=1.0` |
| mixed precision | BF16 autocast when CUDA is available |

## Current Status

v0_baseline 训练已完成（48 epoch）。epoch 20 为 valid_loss 全局最低。选择 `checkpoint_epoch_20.pt` 作为 baseline checkpoint。checkpoint/tokenizer/训练日志/推理日志均已提交。
