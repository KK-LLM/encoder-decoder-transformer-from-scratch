# Encoder-Decoder Transformer v1_curriculum

`v1_curriculum` 是手写 Encoder-Decoder Transformer 英译中项目的课程学习版本，当前已补充 Stage1 到 Stage5 的训练记录、最终 checkpoint、推理日志和推理分析。

该版本基于 `v0_baseline` 的训练和推理分析重新设计。`v0_baseline` 已经证明训练闭环可以跑通，但也暴露出数据语体单一、技术术语不稳定、日常英语泛化不足、复杂逻辑句处理较弱、checkpoint 不支持完整续训等问题。`v1_curriculum` 的目标是用更系统的数据工程和分阶段训练流程来解决这些问题。

## Current Status

| 项目 | 状态 |
|------|------|
| Stage1-Stage5 training data | 已准备 |
| Unified tokenizer | 已准备，48,000 vocab |
| Unified validation set | 已准备，`final_eval.jsonl` |
| Training script | 已准备 |
| Training report | 已补充 |
| V0/V1 comparison | 已补充 |
| Training metrics summary | 已补充 |
| Inference script | 已补充 |
| Inference logs | 已补充 |
| Inference report | 已补充 |
| Checkpoint | 已补充分片文件，Git LFS |
| Final model analysis | 已补充阶段性推理分析 |

当前版本已提交训练数据、tokenizer、训练脚本、最终 checkpoint、推理脚本、推理日志和推理分析文档。

## Training Documents

V1 训练分析文档统一放在 `doc/` 目录下：

| 文件 | 内容 |
|------|------|
| [`doc/architecture_notes.md`](./doc/architecture_notes.md) | V1 当前训练脚本中的模型结构、mask、attention、训练 forward 和架构限制说明 |
| [`doc/v1_curriculum_training_report.md`](./doc/v1_curriculum_training_report.md) | V1 课程学习主训练报告，包含 Stage1 到 Stage5 训练路线、Stage5 参数调整和当前不足 |
| [`doc/v1_vs_v0_comparison.md`](./doc/v1_vs_v0_comparison.md) | V0 baseline 与 V1 curriculum 的训练目标、数据组织和翻译能力对比 |
| [`doc/v1_training_metrics_summary.md`](./doc/v1_training_metrics_summary.md) | V1 课程学习训练记录与指标分析，包含阶段指标、checkpoint 传递和 Stage5 参数对照 |

## Inference Documents

V1 推理相关文件按用途分开放在 `checkpoints/`、`logs/` 和 `results/` 目录下：

| 路径 | 内容 |
|---|---|
| [`checkpoints/README.md`](./checkpoints/README.md) | V1 final best checkpoint 说明 |
| [`checkpoints/checkpoint_best.part00.pt`](./checkpoints/checkpoint_best.part00.pt) | V1 最终 checkpoint 第 1 个分片，使用 Git LFS 管理 |
| [`checkpoints/checkpoint_best.part01.pt`](./checkpoints/checkpoint_best.part01.pt) | V1 最终 checkpoint 第 2 个分片，使用 Git LFS 管理 |
| [`logs/v1_inference_cases.txt`](./logs/v1_inference_cases.txt) | 本轮推理实际使用的 80 条英文输入 |
| [`logs/v1_inference_raw_output.log`](./logs/v1_inference_raw_output.log) | 推理脚本原始控制台输出 |
| [`logs/v1_inference_log_summary.md`](./logs/v1_inference_log_summary.md) | 推理日志结构化摘要 |
| [`results/v1_inference_report.md`](./results/v1_inference_report.md) | V1 完整推理报告 |
| [`results/v1_inference_summary.md`](./results/v1_inference_summary.md) | V1 推理摘要 |

## Goal

本版本不追求工业级通用翻译系统，而是展示一个可复现的课程学习训练流程，让手写 Encoder-Decoder Transformer 从基础英中对齐逐步过渡到复杂能力：

1. 基础日常英译中能力
2. 普通书面句英译中能力
3. because / if / when / although / however 等逻辑结构翻译能力
4. Transformer / tokenizer / encoder / decoder / checkpoint / learning rate 等技术术语稳定翻译能力
5. general_logic / complex_logic 复杂句结构保持能力
6. mixed_regression 防遗忘能力
7. anti-pollution stability，降低规则说明句、模板句和泛泛中文输出倾向

## Curriculum Design

v1 使用 Stage1 到 Stage5 的课程学习训练顺序：

```text
Stage1 -> Stage2 -> Stage3 -> Stage4 -> Stage5
```

每一阶段的定位：

| Stage | 定位 | 训练重点 |
|------|------|------|
| Stage1 | 基础短句阶段 | 建立基础词序、人物、动作、物体、时间、地点和简单日常句映射 |
| Stage2 | 基础能力扩展阶段 | 扩展中等长度日常句、普通书面句、简单逻辑句和基础错误回归 |
| Stage3 | 技术句与术语稳定阶段 | 学习 ML / DL / Transformer / tokenizer / checkpoint 等技术语境 |
| Stage4 | 综合逻辑强化阶段 | 加强普通逻辑、复杂逻辑、技术逻辑混合句和 mixed regression |
| Stage5 | 最终收束阶段 | 综合回放、防遗忘、术语稳定、复杂逻辑、anti-pollution 和最终平衡 |

## Data Scale

| 数据 | 行数 | 用途 |
|------|------:|------|
| `stage1/train.jsonl` | 107,000 | Stage1 训练 |
| `stage2/train.jsonl` | 281,681 | Stage2 训练 |
| `stage3/train.jsonl` | 432,987 | Stage3 训练 |
| `stage4/train.jsonl` | 503,000 | Stage4 训练 |
| `stage5/train.jsonl` | 548,084 | Stage5 训练 |
| `final_eval.jsonl` | 15,000 | Stage1-Stage5 统一验证集 |

每个 stage 的本地目录中可能存在 `test.jsonl`，但这些文件不属于本次 GitHub 更新范围，也不用于本次分阶段训练的 valid loss 计算。v1 训练统一使用 `final_eval.jsonl` 作为验证集，避免不同阶段使用不同验证集导致 valid loss 不可比较。

## Data Sources and Construction

v1 的课程数据由本地 `available_data_unified/` 中的多来源开源英中数据构造而来。该目录整理了通用平行语料、字幕语料、软件本地化语料、技术文档语料、教育类语料、benchmark 数据和术语/词典资源。

这些开源数据没有被直接混合进训练集，而是经过：

1. 数据下载与整理
2. 格式统一
3. 扁平化处理
4. 英中样本筛选
5. 清洗与去重
6. 语体和能力类别划分
7. stage-specific 数据构造
8. replay / error regression / anti-pollution 样本设计

最终形成 Stage1 到 Stage5 的课程学习数据。v1 的重点不只是训练更大的模型，也包括展示从数据收集、数据清洗到训练课程设计的完整数据工程能力。

## Tokenizer

Stage1 到 Stage5 使用同一个 tokenizer：

```text
tokenizer/en_zh_stage_tokenizer_48000/
```

该 tokenizer 使用当前 Stage1 到 Stage5 数据构造，词表大小为 48,000。训练脚本会校验：

| token | expected id |
|------|-------------:|
| `[PAD]` | 0 |
| `[UNK]` | 1 |
| `[CLS]` / BOS | 2 |
| `[SEP]` / EOS | 3 |
| `[MASK]` | 4 |

该 tokenizer 与 `v1_curriculum` 配套使用，不应与 `v0_baseline` 的 16,000 vocab tokenizer 混用。

## Model Configuration

当前训练脚本中的主要模型配置如下：

| 参数 | 值 |
|------|----|
| architecture | Transformer Encoder-Decoder, Post-LN |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_heads` | 12 |
| `num_layers` | 10 |
| `dropout` | 0.08 |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| tokenizer vocab size | 48000 |
| batch size | 168 |
| stage train epochs | 20 |
| optimizer | Adam, `lr=1.0`, `betas=(0.9, 0.98)`, `eps=1e-9` |
| scheduler | LambdaLR + Noam formula |
| warmup steps | 7000 |
| label smoothing | 0.03 |
| mixed precision | BF16 autocast on CUDA |

具体配置以 `src/train_encoder_decoder_curriculum.py` 为准。

## Checkpoint Strategy

Stage1 从随机初始化开始训练。Stage2 到 Stage5 从上一个 stage 的 best checkpoint 继续训练。

保存策略：

- 每个 epoch 保存 `checkpoint_latest.pt`，用于中断恢复
- 每 4 个 epoch 额外保存一次归档 checkpoint
- 当前 run 的 valid loss 刷新最低值时保存 `best_checkpoint.pt`
- 阶段传递使用 `best_checkpoint.pt`，不是简单使用最后一个 epoch
- Stage5 结束后，最终 checkpoint 需要结合 valid loss 和固定样例翻译表现选择

v1 checkpoint 会保存：

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- RNG state
- `training_config`
- train / valid loss history
- `global_step`
- tokenizer 路径和特殊 token id

这与 `v0_baseline` 只保存 `epoch` 和 `model_state_dict` 的策略不同，v1 支持更完整的训练恢复。

## Training Script

训练脚本：

```text
src/train_encoder_decoder_curriculum.py
```

该脚本负责：

1. 加载指定 stage 的 `train.jsonl`
2. 加载统一验证集 `final_eval.jsonl`
3. 加载预构建 48k tokenizer
4. 构建手写 Encoder-Decoder Transformer
5. 使用 Adam + Noam learning rate 训练
6. 每个 epoch 计算统一 valid loss
7. 保存 latest / archive / best checkpoint
8. 打印基础句、逻辑句、技术术语、复杂逻辑和 anti-pollution 固定样例翻译

## Inference Script

推理脚本：

```text
src/infer_encoder_decoder_curriculum.py
```

从仓库根目录运行：

```bash
cat encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.part00.pt \
    encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.part01.pt \
    > encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.pt

python3 encoder_decoder/v1_curriculum/src/infer_encoder_decoder_curriculum.py \
  --checkpoint encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.pt \
  --tokenizer-dir encoder_decoder/v1_curriculum/tokenizer/en_zh_stage_tokenizer_48000 \
  --input-file encoder_decoder/v1_curriculum/logs/v1_inference_cases.txt \
  --batch-size 16
```

当前推理脚本使用 batch greedy decode。推理报告中的结论来自 `logs/v1_inference_raw_output.log`。

## Run

从仓库根目录运行：

```bash
bash scripts/run_train_encoder_decoder_v1_curriculum.sh
```

如果直接运行 Python 脚本，推荐显式传入 stage、输出目录和 resume checkpoint。具体参数以训练脚本为准。

## Current Limitations

1. 当前不提交 stage 下的 `test.jsonl`，统一验证只使用 `final_eval.jsonl`。
2. 当前版本仍以内嵌模型源码为主，尚未拆分为独立 `model.py`。
3. 当前推理结果显示，复杂逻辑、普通书面句、source following 和 anti-pollution 仍然没有完全稳定。
4. 当前模型仍保留 Post-LN、LayerNorm、ReLU FFN 和 greedy decode 等基础设计，后续仍有结构优化空间。

## Next Updates

后续将继续补充：

1. 更细的错误分析和泛化分析
2. 模型结构和推理策略的后续优化记录
3. 下一版结构改进后的对照训练和推理结果
