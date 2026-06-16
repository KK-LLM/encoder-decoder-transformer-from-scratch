# Encoder-Decoder v0_baseline

手写 Transformer Encoder-Decoder 的 baseline 版本，使用 Helsinki-NLP/opus-100 (en-zh) 数据集训练英文到中文的机器翻译模型。

## 任务

英文 → 中文单向机器翻译（source_lang="en", target_lang="zh"）

## 模型架构

原始 Transformer Encoder-Decoder（Post-LN），所有模块从零手写，不依赖 `nn.Transformer`。

### 模型配置

| 参数 | 值 |
|------|-----|
| d_model | 512 |
| d_ff | 2048 |
| num_heads | 8 |
| num_layers | 8 (encoder 和 decoder 各 8 层) |
| dropout | 0.1 |
| max_src_len | 96 |
| max_tgt_len | 96 |
| vocab_size | 16000 |

模型源码内嵌在训练和推理脚本中，关键模块包括：

- TokenEmbedding + PositionalEncoding
- MultiHeadAttention（含 split/combine heads）
- Post-LN Residual Connection
- PositionwiseFeedForward
- Encoder / Decoder / Generator
- EncoderDecoderTransformer（顶层封装）

详见 [架构说明文档](../../../docs/architecture_notes.md)。

## 数据

| 项目 | 说明 |
|------|------|
| 来源 | [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) 的 en-zh 子集 |
| 格式 | HuggingFace Arrow (Translation 类型) |
| 原始规模 | train 1,000,000 / validation 2,000 / test 2,000 |
| 本次使用 | train 前 500,000 → 实际有效 ~499,989 / valid 2,000 |
| 样例 | [sample_data.jsonl](./data/sample_data.jsonl)（200 条） |
| 完整数据 | [data/opus100_en_zh_local/](./data/opus100_en_zh_local/)（需 Git LFS） |

数据清洗方式（TranslationDataset 内自动处理）：
- 过滤空字符串样本
- 过滤 BOS + 正文 + EOS 后 token 数 < 3 的样本

详见 [数据说明](./data/README.md)。

## Tokenizer

| 项目 | 说明 |
|------|------|
| 类型 | BPE（Fast tokenizer） |
| 底座 | bert-base-multilingual-cased |
| vocab_size | 16000 |
| 共用 | 源语言和目标语言共用一个 tokenizer |
| 训练语料 | 与训练集相同的中英平行语料 |

特殊 token：

| Token | ID |
|-------|-----|
| PAD ([PAD]) | 0 |
| BOS ([CLS]) | 2 |
| EOS ([SEP]) | 3 |

详见 [tokenizer 说明](./tokenizer/README.md)。

## 训练配置

| 参数 | 值 |
|------|-----|
| batch_size | 280 |
| epochs | 48 |
| optimizer | Adam (lr=1.0, betas=(0.9, 0.98), eps=1e-9) |
| learning rate | Noam 公式：lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5)) |
| scheduler | LambdaLR + Noam |
| warmup_steps | 3000 |
| loss function | CrossEntropyLoss (ignore_index=PAD_ID) |
| label_smoothing | 0.1 |
| gradient clipping | max_norm=1.0 |
| mixed precision | BF16 (torch.bfloat16 autocast) |
| device | NVIDIA RTX 5090 |

### Checkpoint 策略

- 每 4 个 epoch 保存一次（epoch % 4 == 0）
- 训练结束后保存 final model
- 当前版本不支持 checkpoint 自动续训（中断后需从头开始）
- 本轮不提交权重文件，详见 [checkpoints/README.md](./checkpoints/README.md)

## 推理

推理脚本支持：

- 加载训练时保存的 tokenizer
- 加载 checkpoint 权重
- 构造 src_mask / tgt_mask
- Greedy decode 自回归生成
- 固定样例 + 交互式输入

当前仅实现 greedy decode，后续计划增加 beam search。

## 当前训练状态

**训练正在进行中。** 本轮 GitHub 提交侧重展示项目结构、代码实现和训练流程，暂不包含：

- 最终训练日志
- 推理样例或评估指标
- Checkpoint 权重文件

训练完成后将更新实验结果和推理样例。

## Limitations（当前版本限制）

1. 本版本为 v0_baseline，侧重展示从零实现 Encoder-Decoder 的完整闭环，不宣称达到最优翻译效果
2. 模型源码内嵌在 train/infer 脚本中，尚未抽离为独立 model.py
3. 推理仅支持 greedy decode，无 beam search
4. 无 checkpoint 自动续训逻辑
5. 无 BLEU / chrF 等自动评估指标（待后续版本补充）
6. 训练策略、数据清洗、评估指标仍需进一步系统化

## Next Steps

1. 补充完整训练日志摘要和 BLEU / chrF 评估
2. 增加典型推理样例和错误案例分析
3. 增加 beam search 解码策略
4. 增加 checkpoint 加载和自动续训能力
5. 优化 tokenizer 和数据清洗策略
6. 增加 config.yaml 配置系统
7. 抽离 model.py 实现模块化
8. 推进 v1_improved 版本

## 如何运行

### 训练

```bash
cd projects/encoder_decoder/v0_baseline/src
python train_encoder_decoder_opus.py
```

### 推理

修改 `infer_encoder_decoder.py` 中的 `MODEL_DIR` 和 `MODEL_PATH` 指向训练输出的 checkpoint 目录，然后：

```bash
cd projects/encoder_decoder/v0_baseline/src
python infer_encoder_decoder.py
```
