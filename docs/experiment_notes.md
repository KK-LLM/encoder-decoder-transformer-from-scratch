# 实验记录

## v0_baseline — opus-100 en-zh

### 实验目的

1. 端到端验证手写 Encoder-Decoder Transformer 的训练、验证、推理完整闭环，确保 pipeline 无阻塞性缺陷
2. 在单一、相对干净的中英数据集上初步观察多 epoch 训练下的翻译质量趋势和学习曲线

### 关键配置

| 参数 | 值 |
|------|-----|
| 数据集 | Helsinki-NLP/opus-100 (en-zh) |
| 训练样本 | 500,000（有效 ~499,989） |
| 模型参数 | d_model=512, d_ff=2048, heads=8, layers=8 |
| vocab_size | 16000 |
| batch_size | 280 |
| epochs | 48 |
| warmup_steps | 3000 |
| optimizer | Adam (lr=1.0, betas=(0.9, 0.98), eps=1e-9) |
| lr schedule | Noam (warmup + inverse sqrt decay) |
| label_smoothing | 0.1 |
| mixed precision | BF16 |
| GPU | NVIDIA RTX 5090 |

### 训练状态

**当前训练仍在进行中。** 训练完成后将在此文件中补充：

- 训练 loss 曲线
- 验证 loss 曲线
- BLEU / chrF 评估结果
- 典型推理样例和错误分析
