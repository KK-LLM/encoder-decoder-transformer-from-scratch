# Tokenizer

本版本的训练脚本使用 `tokenizer/bert_base_multilingual_cased_tokenizer/` 作为 base tokenizer snapshot，然后在 OPUS-100 en-zh 训练语料上重新训练一个 16000 vocab tokenizer。

## Runtime Behavior

训练脚本中的 tokenizer 流程：

1. `AutoTokenizer.from_pretrained(...)` 加载本地 base tokenizer。
2. 从 train split 中读取英文和中文文本。
3. 调用 `train_new_from_iterator(..., vocab_size=16000)` 重新训练 tokenizer。
4. 源语言和目标语言共用同一个 tokenizer。
5. `PAD_ID`、`BOS_ID`、`EOS_ID` 从训练后的 tokenizer 动态读取。
6. checkpoint 保存时同步保存训练后的 tokenizer 到训练输出目录。

BOS 使用 `[CLS]`，EOS 使用 `[SEP]`。实际 token id 以训练脚本运行时打印的 tokenizer 为准。

## Uploaded Files

### Base Tokenizer Snapshot

供训练脚本从本地加载并重新训练 16000 vocab tokenizer：

```text
bert_base_multilingual_cased_tokenizer/
├── tokenizer.json
└── tokenizer_config.json
```

### Trained Tokenizer

v0_baseline 训练完成后生成的 16000 vocab tokenizer，与 `checkpoints/checkpoint_epoch_20.pt` 配套用于推理：

```text
trained_tokenizer_16000/
├── tokenizer.json
└── tokenizer_config.json
```

该 tokenizer 的 vocab_size=16000，PAD_ID=0，BOS_ID=2，EOS_ID=3。不可用 base tokenizer snapshot 直接替代。
