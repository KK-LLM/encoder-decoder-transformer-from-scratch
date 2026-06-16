# Encoder-Decoder Transformer v0_baseline

`v0_baseline` 是手写 Transformer Encoder-Decoder 的 baseline 版本，任务是英文到中文单向机器翻译。该版本重点展示模型源码、训练闭环、推理闭环、tokenizer 训练与 checkpoint 保存机制。

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
| [src/infer_encoder_decoder.py](./src/infer_encoder_decoder.py) | 手写模型源码 + 推理闭环 |
| [data/sample_data.jsonl](./data/sample_data.jsonl) | 200 条真实中英样例 |
| [tokenizer/README.md](./tokenizer/README.md) | tokenizer 使用说明 |
| [checkpoints/README.md](./checkpoints/README.md) | checkpoint 保存与当前限制 |

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

当前 checkpoint 只包含：

```python
{
    "epoch": epoch,
    "model_state_dict": cpu_state_dict,
}
```

当前脚本不保存 optimizer / scheduler 状态，不支持自动续训。本轮不提交真实权重文件。

## Inference

推理脚本支持加载训练输出目录中的 tokenizer 和 checkpoint，构造 `src_mask` / `tgt_mask`，并执行 greedy decode。由于训练仍在进行中，本轮不提交推理日志、推理样例或效果结论。

## Current Training Status

当前训练仍在进行中。本轮提交只展示 baseline 版本的结构、代码和说明，不写最终训练结果、不写最佳 epoch、不写 best valid loss、不写 BLEU / chrF、不写推理效果结论。

## Limitations

1. 当前版本是 `v0_baseline`，主要用于展示完整 Encoder-Decoder Transformer pipeline。
2. 当前训练仍在进行中，本轮不提交训练日志、推理日志、推理样例和 checkpoint。
3. 当前不提供最终训练指标和翻译质量结论。
4. 推理脚本当前使用 greedy decode。
5. 模型源码内嵌在 train / infer 脚本中，未拆分为独立 `model.py`。
6. checkpoint 当前只保存 `epoch` 和 `model_state_dict`，不保存 optimizer / scheduler 状态，不支持自动续训。
7. 当前版本更强调完整 pipeline 和源码可读性，而不是最终翻译效果最优。

## Run

从仓库根目录运行训练：

```bash
bash scripts/run_train_encoder_decoder.sh
```

训练产生 checkpoint 后运行推理：

```bash
bash scripts/run_infer_encoder_decoder.sh
```
