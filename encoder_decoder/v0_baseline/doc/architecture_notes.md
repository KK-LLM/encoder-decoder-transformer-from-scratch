# Architecture Notes

本文记录当前 Encoder-Decoder Transformer v0_baseline 的模型结构。模型源码内嵌在训练脚本和推理脚本中，未拆分为独立 `model.py`。

## Overall Architecture

当前模型是原始 Transformer Encoder-Decoder 架构，使用 Post-LN residual connection：

```text
output = LayerNorm(residual + Dropout(sublayer_out))
```

## Main Components

| 模块 | 说明 |
|------|------|
| `TokenEmbedding` | token id embedding，并乘以 `sqrt(d_model)` |
| `PositionalEncoding` | sinusoidal positional encoding |
| `LayerNorm` | 手写 layer normalization |
| `PostNormResidualConnection` | Post-LN residual connection |
| `MultiHeadAttention` | 手写 Q/K/V projection、split heads、scaled dot-product attention、combine heads |
| `PositionwiseFeedForward` | 两层 FFN + ReLU + dropout |
| `EncoderLayer` | self-attention + FFN |
| `DecoderLayer` | masked self-attention + cross-attention + FFN |
| `Encoder` | 多层 encoder stack |
| `Decoder` | 多层 decoder stack |
| `Generator` | decoder hidden states 到 vocabulary logits |
| `EncoderDecoderTransformer` | encode / decode / forward 顶层封装 |

## Masks

`src_mask` 用于屏蔽 source padding token：

```text
src_mask: [B, 1, 1, S]
```

`tgt_mask` 同时包含 target padding mask 和 causal mask：

```text
tgt_mask: [B, 1, T, T]
```

causal mask 使用下三角矩阵，保证 decoder 当前位置不能看到未来 token。

## Training Forward

训练时 target 会拆成：

```text
tgt_in  = tgt[:, :-1]
tgt_out = tgt[:, 1:]
```

模型输出 logits 后，对 `tgt_out` 计算 cross entropy loss，并使用 `PAD_ID` 作为 ignore index。

## Greedy Decode

推理脚本当前实现 greedy decode：

1. 编码 source sentence。
2. decoder 从 BOS token 开始。
3. 每一步构造 causal `tgt_mask`。
4. 取最后一个位置 logits 的 argmax 作为 next token。
5. 遇到 EOS 或达到最大长度后停止。

当前版本不提交推理样例或翻译质量结论。
