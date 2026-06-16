# Experiment Notes

本文只记录当前 `v0_baseline` 已确定的信息。当前训练仍在进行中，本轮不提交最终训练结果、完整训练日志、推理样例或权重文件。

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

当前训练仍在进行中。训练脚本已经支持 train / valid loss 计算和 checkpoint 保存。本轮暂不提交完整训练日志、推理样例和权重文件。

训练完成后的真实结果应按实际日志和推理表现补充到对应版本目录中。
