# Training Log Summary

## Overview

本轮实验是手写 Encoder-Decoder Transformer 在 OPUS100 en-zh 数据集上的 baseline train run。

- **任务**：English → Chinese 翻译
- **数据**：OPUS100 en-zh 本地子集
- **模型**：手写 Encoder-Decoder Transformer（Post-Norm, Sinusoidal PE, Multi-Head Attention）
- **版本**：baseline / first training run
- **目标**：验证模型、tokenizer、训练循环、验证循环、checkpoint 保存和固定例句推理的完整闭环

---

## Training Script

训练脚本：`Encoder-Decoder翻译训练-opus.py`

脚本包含以下完整训练逻辑：

- **模型构建**：`make_transformer()` 构建 Encoder-Decoder Transformer，包含 Token Embedding、Sinusoidal Positional Encoding、Multi-Head Attention、Positionwise FFN、Post-Norm Residual Connection
- **Tokenizer 训练**：基于 `bert_base_multilingual_cased_tokenizer` 在训练语料上重新训练小词表 tokenizer（vocab_size=16000）
- **数据加载**：从 `opus100_en_zh_local` 加载 HuggingFace Dataset，构造 `TranslationDataset` 和 `DataLoader`
- **训练与验证**：Noam learning rate scheduler + Adam optimizer + label smoothing + mixed precision (bfloat16)
- **固定例句推理**：每个 epoch 结束后 greedy decode 12 条固定英文句子并打印翻译结果
- **Checkpoint 保存**：每 4 个 epoch 保存一次 checkpoint
- **Final model 保存**：训练结束后保存 final_model.pt 和 tokenizer

### Config Note

当前训练脚本中的默认配置与训练日志中的实际运行配置**一致**（num_epochs=48, d_model=512, d_ff=2048, heads=8, layers=8, batch_size=280 等），未发现不一致。

---

## Dataset

| 属性 | 值 |
|---|---|
| 数据集名称 | OPUS100 en-zh |
| 语言方向 | English → Chinese |
| train raw samples | 500,000 |
| valid raw samples | 2,000 |
| train tokenized kept samples | 499,980 |
| valid tokenized kept samples | 2,000 |
| max_src_len | 96 |
| max_tgt_len | 96 |

tokenization 过程中仅过滤了极少数空文本或过短样本（< 3 tokens），数据保留率约 99.996%。

---

## Tokenizer

| 项目 | 值 |
|---|---|
| base tokenizer | `./bert_base_multilingual_cased_tokenizer` (fast) |
| 是否重新训练 | 是，基于中英文训练语料 |
| new tokenizer vocab_size | 16,000 |
| PAD_ID | 0 |
| BOS_ID | 2 (CLS token) |
| EOS_ID | 3 (SEP token) |

这是一个针对中英文翻译任务重新训练的小词表 tokenizer，相比原始 BERT multilingual tokenizer 的 119k 词表更紧凑，适合小规模模型实验。

---

## Model Config

| 参数 | 值 |
|---|---|
| d_model | 512 |
| d_ff | 2,048 |
| num_heads | 8 |
| num_layers | 8 (encoder + decoder 各 8 层) |
| dropout | 0.1 |
| vocab_size | 16,000 |
| batch_size | 280 |
| max_len | (96, 96) |

以上配置均以训练日志中的实际运行值为准。

---

## Training Setup

| 项目 | 值 |
|---|---|
| optimizer | Adam (lr=1.0, betas=(0.9, 0.98), eps=1e-9) |
| scheduler | Noam (warmup_steps=3000, factor=1.0) |
| warmup_steps | 3000 |
| label_smoothing | 0.1 |
| mixed precision | AMP enabled, dtype=bfloat16 |
| device | NVIDIA GeForce RTX 5090 (CUDA) |
| gradient clipping | max_norm=1.0 |
| checkpoint interval | every 4 epochs |
| fixed examples | 12 sentences after each epoch |
| total epochs | 48 |
| steps per epoch | 1,786 |

---

## Metrics Summary

| Epoch | Train Loss | Valid Loss | LR | Checkpoint | Note |
|---|---|---|---|---|---|
| 1 | 5.9811 | 4.5595 | 0.000480 | | early training |
| 4 | 3.8329 | 3.7286 | 0.000523 | ✓ | |
| 8 | 3.5438 | 3.5457 | 0.000370 | ✓ | train/valid loss 接近 |
| 12 | 3.4021 | 3.4938 | 0.000302 | ✓ | |
| 16 | 3.3113 | 3.4777 | 0.000261 | ✓ | valid loss 开始震荡 |
| 20 | 3.2443 | 3.4693 | 0.000234 | ✓ | **Best valid loss** |
| 24 | 3.1911 | 3.4914 | 0.000213 | ✓ | 过拟合趋势明显 |
| 28 | 3.1469 | 3.5013 | 0.000198 | ✓ | |
| 32 | 3.1097 | 3.5089 | 0.000185 | ✓ | |
| 36 | 3.0765 | 3.5307 | 0.000174 | ✓ | |
| 40 | 3.0475 | 3.5418 | 0.000165 | ✓ | |
| 44 | 3.0214 | 3.5682 | 0.000158 | ✓ | valid loss 持续恶化 |
| 48 | 2.9984 | 3.5653 | 0.000151 | ✓ | final epoch |

---

## Best Checkpoint

本轮训练中，验证集 loss 在 **epoch 20** 达到最低值 **3.4693**。

**推荐 checkpoint**：`checkpoint_epoch_20.pt`

推荐理由：
- epoch 20 是 valid_loss 的全局最低点
- 在此之后 train_loss 继续下降但 valid_loss 波动上升，呈现典型过拟合趋势
- 后期 epoch（30+）的 valid_loss 明显高于 epoch 20，泛化能力更差

---

## Loss Trend Analysis

1. **Early stage (epoch 1-5)**：train_loss 从 5.98 快速下降至 3.73，valid_loss 从 4.56 降至 3.65，模型开始学习基本翻译模式，Noam scheduler 处于 warmup → decay 过渡期。

2. **Middle stage (epoch 6-15)**：train_loss 和 valid_loss 同步下降但速度放缓。epoch 15 时 valid_loss 达到 3.4750，接近平台期。

3. **Best epoch (epoch 16-20)**：train_loss 继续下降，valid_loss 在 epoch 20 降至全局最低 3.4693。此后 valid_loss 开始反弹。

4. **Overfitting phase (epoch 20-48)**：train_loss 持续下降至 2.9984（epoch 48），但 valid_loss 从 3.4693 波动上升至 3.5653。train/valid loss 之间的 gap 从 ~0.23 扩大至 ~0.57，说明模型在训练集上持续拟合但在验证集上泛化变差。

5. **Later epochs (epoch 34-48)**：valid_loss 维持在 3.52-3.57 区间，部分 epoch 出现 spike（如 epoch 42: 3.5585, epoch 44: 3.5682），训练后期稳定性下降。

---

## Checkpoint Strategy

本轮训练每 **4 个 epoch** 保存一次 checkpoint，共保存 12 个 checkpoint：

| Epoch | Checkpoint File |
|---|---|
| 4 | checkpoint_epoch_4.pt |
| 8 | checkpoint_epoch_8.pt |
| 12 | checkpoint_epoch_12.pt |
| 16 | checkpoint_epoch_16.pt |
| 20 | checkpoint_epoch_20.pt |
| 24 | checkpoint_epoch_24.pt |
| 28 | checkpoint_epoch_28.pt |
| 32 | checkpoint_epoch_32.pt |
| 36 | checkpoint_epoch_36.pt |
| 40 | checkpoint_epoch_40.pt |
| 44 | checkpoint_epoch_44.pt |
| 48 | checkpoint_epoch_48.pt |

此外，训练结束后保存了 `final_model.pt`（epoch 48 状态）。

**GitHub 推荐做法**：仅上传 `checkpoint_epoch_20.pt`（best checkpoint），其余 checkpoint 通过本日志记录即可。如权重文件较大，不建议全部上传。

---

## Limitations

- **评测指标单一**：当前仅使用 valid loss 作为主要量化指标，尚未加入 BLEU、chrF、sacreBLEU 等翻译质量自动评测指标
- **固定例句仅为人工观察**：12 条固定例句的 greedy decoding 输出仅用于人工定性观察训练趋势，不能替代系统评测
- **Greedy decoding 局限**：greedy search 容易导致输出偏短、重复 token、语义偏移或幻觉
- **过拟合趋势**：epoch 20 之后 valid loss 明显上升，说明当前 baseline 在约 20 epoch 后开始过拟合，缺乏 early stopping 机制
- **数据清洗有限**：仅过滤了空文本和过短样本（< 3 tokens），未做进一步的双语质量、长度比、语言检测等清洗
- **配置管理**：训练参数硬编码在脚本中，不利于实验追踪和复现

---

## 完整原始日志

完整训练日志保存在: `logs/opus_train_log.log`

推荐优先阅读本文件（`train_log_summary.md`）和 `train_metrics.csv`，原始日志主要用于复查完整训练输出、固定例句翻译细节和 checkpoint 保存记录。
