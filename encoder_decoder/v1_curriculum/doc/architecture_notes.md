# Architecture Notes

本文记录 `v1_curriculum` 当前训练脚本中的模型结构。V1 仍然是手写 Encoder-Decoder Transformer，模型源码内嵌在 `src/train_encoder_decoder_curriculum.py`，还没有拆分为独立的 `model.py`。

这份说明只描述当前已经提交的 V1 实现，不写后续还没有合并进仓库的结构改动。

## Overall Architecture

当前模型仍采用原始 Transformer Encoder-Decoder 主体结构，Encoder 负责读取英文 source sentence，Decoder 基于 encoder memory 和已生成的 target prefix 预测下一个中文 token。

残差结构使用 Post-LN：

```text
output = LayerNorm(residual + Dropout(sublayer_out))
```

这和 V0 的主结构保持一致，但 V1 在模型规模、tokenizer、训练数据组织、checkpoint 保存和阶段训练流程上都做了升级。

## Main Configuration

| 配置项 | 当前设置 |
|------|------|
| task | English-to-Chinese translation |
| architecture | Transformer Encoder-Decoder, Post-LN |
| tokenizer | `en_zh_stage_tokenizer_48000` |
| vocab size | 48,000 |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_heads` | 12 |
| `d_head` | 64 |
| `num_layers` | 10 encoder layers + 10 decoder layers |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| default dropout | 0.08 |
| default label smoothing | 0.03 |
| default warmup steps | 7000 |

Stage5 最终收束训练中对部分训练超参数做过调整，具体分析放在 V1 训练报告中。这里的表格只记录当前仓库训练脚本的默认结构和默认配置。

## Main Components

| 模块 | 当前实现 |
|------|------|
| `TokenEmbedding` | token id embedding，并乘以 `sqrt(d_model)` 做尺度调整 |
| `PositionalEncoding` | 固定 sinusoidal positional encoding，通过 buffer 保存，不参与训练更新 |
| `LayerNorm` | 手写 LayerNorm，包含 `gamma` 和 `beta` 参数 |
| `PostNormResidualConnection` | `residual + dropout(sublayer_out)` 后再做 LayerNorm |
| `MultiHeadAttention` | 手写 Q/K/V projection、split heads、scaled dot-product attention、combine heads 和 output projection |
| `PositionwiseFeedForward` | `Linear(d_model, d_ff) -> ReLU -> Dropout -> Linear(d_ff, d_model)` |
| `EncoderLayer` | self-attention + FFN |
| `DecoderLayer` | masked self-attention + cross-attention + FFN |
| `Encoder` | 10 层 encoder stack |
| `Decoder` | 10 层 decoder stack |
| `Generator` | `Linear(d_model, vocab_size)` 输出 vocabulary logits |
| `EncoderDecoderTransformer` | 封装 `encode`、`decode` 和 `forward` |

当前 V1 没有使用 `nn.Transformer` 封装，也没有把 attention、FFN、mask 和训练 forward 隐藏到高层库里。

## Masks

`src_mask` 用于屏蔽 source padding token：

```text
src_mask: [B, 1, 1, S]
```

`tgt_mask` 同时包含 target padding mask 和 causal mask：

```text
tgt_mask: [B, 1, T, T]
```

mask 中 `True` 表示当前位置可以被 attention 看到，`False` 表示需要屏蔽。causal mask 使用下三角矩阵，保证 decoder 当前位置不能看到未来 token。

## Attention Flow

V1 的 multi-head attention 仍然是普通 scaled dot-product attention：

```text
scores = QK^T / sqrt(d_head)
scores = masked_fill(scores, -1e9)
attention_weights = softmax(scores)
context = attention_weights @ V
```

随后将多个 head 合并，再经过 `out_proj` 回到 `d_model` 维度。

## Training Forward

训练时 target 会拆成：

```text
tgt_in  = tgt[:, :-1]
tgt_out = tgt[:, 1:]
```

模型接收 `src`、`tgt_in`、`src_mask` 和 `tgt_mask`，输出 `[B, T, vocab_size]` logits。loss 使用 `F.cross_entropy`，并通过 `ignore_index=PAD_ID` 忽略 padding token。

训练脚本中还保留了以下训练稳定性处理：

- Adam optimizer，`betas=(0.9, 0.98)`，`eps=1e-9`
- Noam learning rate
- label smoothing
- gradient clipping，`max_norm=1.0`
- CUDA BF16 autocast
- `checkpoint_latest.pt`、`checkpoint_epoch_{epoch}.pt` 和 `best_checkpoint.pt` 保存策略

## Greedy Translation in Training Logs

训练脚本中的固定样例翻译使用 `greedy_translate`。流程是：

1. 先编码英文 source。
2. decoder 从 BOS token 开始逐步生成。
3. 每一步重新构造 target causal mask。
4. 取最后一个位置 logits 的 argmax 作为 next token。
5. 屏蔽 PAD / BOS，前几个 token 不允许直接生成 EOS。
6. 遇到 EOS 或达到最大生成长度后停止。

当前 V1 训练脚本还没有把 beam search 作为默认推理策略提交进来。

## Difference from V0 Baseline

从模型结构看，V1 不是完全换一套架构，而是在 V0 手写 Transformer 的基础上扩大模型容量并增强训练工程能力：

| 维度 | V0 baseline | V1 curriculum |
|------|------|------|
| tokenizer vocab | 16,000 | 48,000 |
| `d_model` | 512 | 768 |
| `d_ff` | 2048 | 3072 |
| `num_heads` | 8 | 12 |
| encoder / decoder layers | 8 + 8 | 10 + 10 |
| validation | 单一 baseline valid split | 统一 `final_eval.jsonl` |
| checkpoint | `epoch` + `model_state_dict` | model / optimizer / scheduler / RNG / config / loss history |
| training route | 单阶段 baseline | Stage1 到 Stage5 课程学习 |

V1 的主要变化不只是参数量变大，更重要的是把数据组织、阶段训练、checkpoint 传递、固定样例观察和训练诊断串成了一个更完整的训练流程。

## Current Architectural Limitations

从当前结构看，V1 仍然保留了一些基础版本的设计限制：

1. 仍然使用 Post-LN，在 10 层 encoder / decoder 的设置下，深层训练稳定性和最终收束还有优化空间。
2. Encoder stack 和 Decoder stack 末尾没有额外 final norm，最后一层输出分布主要依赖各层内部的 Post-LN 来稳定。
3. FFN 仍然是 ReLU FFN，对长句、复杂逻辑句和技术句的非线性表达能力还有继续提升空间。
4. `Generator` 是独立线性层，当前没有和 target embedding 做权重共享。
5. 当前训练脚本中的样例翻译仍然使用 greedy decoding，对长句和复杂句的搜索空间利用不充分。
6. 模型定义仍然内嵌在训练脚本中，适合展示完整训练闭环，但后续做多版本结构对比时，独立模型模块会更清晰。
