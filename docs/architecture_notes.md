# 模型架构说明

## Encoder-Decoder Transformer (Post-LN)

本项目实现了原始 "Attention Is All You Need" 论文中的 Transformer Encoder-Decoder 架构（Post-LayerNorm 变体）。

### 整体结构

```
Input (src)                          Input (tgt, shifted right)
    |                                      |
[TokenEmbedding]                      [TokenEmbedding]
    |                                      |
[PositionalEncoding]                  [PositionalEncoding]
    |                                      |
[Encoder × N]                         [Decoder × N]
    |                                      ↑
    └────────→ memory ─────────────────────┘
                                               |
                                          [Generator (Linear)]
                                               |
                                          logits → CrossEntropyLoss
```

### Encoder Layer

```
x ──→ MultiHeadAttention(q=x, k=x, v=x, mask=src_mask) ──→ Dropout ──→ + ──→ LayerNorm
      (self-attention)                                              ↑        |
                                                                    |        |
                                                              residual ────→ FFN ──→ Dropout ──→ + ──→ LayerNorm → output
                                                                                                 ↑
                                                                                           residual
```

### Decoder Layer

```
x ──→ Masked MultiHeadAttention(q=x, k=x, v=x, mask=tgt_mask) ──→ Dropout ──→ + ──→ LayerNorm
      (self-attention, causal)                                                 ↑        |
                                                                         residual     |
                                                                                      └──→ CrossAttention(q=x, k=memory, v=memory, mask=src_mask)
                                                                                              |
                                                                                         Dropout ──→ + ──→ LayerNorm
                                                                                                      ↑        |
                                                                                                residual     |
                                                                                                             └──→ FFN ──→ Dropout ──→ + ──→ LayerNorm → output
                                                                                                                                      ↑
                                                                                                                                residual
```

### 关键模块

#### MultiHeadAttention

- Q/K/V 线性投影：`nn.Linear(d_model, d_model)`
- Split heads：`[B, L, d_model] → [B, H, L, d_head]`
- Scaled dot-product attention：`softmax(QK^T / √d_head) × V`
- Combine heads + output projection

#### PositionalEncoding

- 正弦/余弦固定位置编码（不可学习参数）
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

#### PostNormResidualConnection

```
output = LayerNorm(residual + Dropout(sublayer_out))
```

与 Pre-LN（LayerNorm 在前）不同，Post-LN 是原论文的默认方式。

### Mask 机制

| Mask | 形状 | 用途 |
|------|------|------|
| src_mask | [B, 1, 1, S] | Encoder self-attention + Decoder cross-attention；屏蔽 PAD token |
| tgt_mask | [B, 1, T, T] | Decoder self-attention；下三角因果掩码 + PAD 掩码 |

### 训练策略

- **Noam Learning Rate**：warmup 阶段线性增长，之后按 step^(-0.5) 衰减
- **Label Smoothing**：0.1，缓解过拟合
- **Gradient Clipping**：max_norm=1.0，稳定训练
- **Mixed Precision**：BF16 autocast，加速训练

### 推理策略

- Greedy decode：每步取 logit 最大的 token
- 禁止生成 PAD / BOS token
- 前 min_new_tokens 步禁止生成 EOS，防止空输出
