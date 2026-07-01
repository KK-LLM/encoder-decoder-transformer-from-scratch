# Encoder-Decoder Transformer from Scratch

> Encoder-Decoder Transformer from scratch learning and training lab.

这个仓库用于整理手写 Encoder-Decoder Transformer 英文到中文翻译项目。当前仓库包含两个连续版本：

- `v0_baseline`：基于 OPUS-100 en-zh 的 baseline 版本，训练已完成，并已提交 checkpoint、训练日志、推理日志和错误分析。
- `v1_curriculum`：基于 `v0_baseline` 经验重新设计的课程学习版本，已补充 Stage1 到 Stage5 训练记录、V0/V1 对比、最终 checkpoint、推理日志和推理分析。

## Project Overview

本项目聚焦原始 Transformer Encoder-Decoder 架构，不依赖 `nn.Transformer` 封装。训练脚本保留完整模型结构与核心训练流程，便于阅读和复现从数据构造、tokenizer、训练、验证到 checkpoint 保存的端到端 pipeline。

当前仓库只包含 Encoder-Decoder Transformer 相关内容。Decoder-only from scratch 后续会单独新建仓库，不属于当前仓库。

## Why This Repo

- 手写 Transformer 基础模块：`TokenEmbedding`、`PositionalEncoding`、`LayerNorm`、`MultiHeadAttention`、`Encoder`、`Decoder`、`Generator`
- 搭建训练闭环：数据加载、tokenizer 使用、mask 构造、loss 计算、Noam learning rate、checkpoint 保存
- 保留版本化实验结构：`v0_baseline` 记录 baseline 训练结果，`v1_curriculum` 记录课程学习训练方案
- 展示数据工程流程：从多来源开源英中数据收集、整理、清洗、去重到分阶段课程数据构造

## Current Progress

| 项目 | 版本 | 状态 |
|------|------|------|
| Encoder-Decoder Transformer | [v0_baseline](./encoder_decoder/v0_baseline/) | 训练已完成，已提交 checkpoint、训练日志和推理分析 |
| Encoder-Decoder Transformer | [v1_curriculum](./encoder_decoder/v1_curriculum/) | 已提交分阶段训练数据、统一 48k tokenizer、训练/推理脚本、最终 checkpoint 和 V1 训练推理分析文档 |

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── requirements.txt
├── encoder_decoder/
│   ├── README.md
│   ├── v0_baseline/
│   │   ├── README.md
│   │   ├── doc/
│   │   │   ├── architecture_notes.md
│   │   │   ├── checkpoint_and_reproducibility.md
│   │   │   └── experiment_notes.md
│   │   ├── src/
│   │   ├── data/
│   │   ├── tokenizer/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── results/
│   └── v1_curriculum/
│       ├── README.md
│       ├── checkpoints/
│       │   ├── README.md
│       │   ├── checkpoint_best.part00.pt
│       │   └── checkpoint_best.part01.pt
│       ├── doc/
│       │   ├── architecture_notes.md
│       │   ├── v1_curriculum_training_report.md
│       │   ├── v1_vs_v0_comparison.md
│       │   └── v1_training_metrics_summary.md
│       ├── src/
│       │   ├── train_encoder_decoder_curriculum.py
│       │   └── infer_encoder_decoder_curriculum.py
│       ├── data/
│       │   ├── README.md
│       │   ├── final_eval.jsonl
│       │   ├── stage1/train.jsonl
│       │   ├── stage2/train.jsonl
│       │   ├── stage3/train.jsonl
│       │   ├── stage4/train.jsonl
│       │   └── stage5/train.jsonl
│       ├── logs/
│       │   ├── README.md
│       │   ├── v1_inference_cases.txt
│       │   ├── v1_inference_log_summary.md
│       │   └── v1_inference_raw_output.log
│       ├── results/
│       │   ├── README.md
│       │   ├── v1_inference_report.md
│       │   └── v1_inference_summary.md
│       └── tokenizer/
│           ├── README.md
│           └── en_zh_stage_tokenizer_48000/
└── scripts/
    ├── run_train_encoder_decoder.sh
    ├── run_infer_encoder_decoder.sh
    └── run_train_encoder_decoder_v1_curriculum.sh
```

## v0_baseline

`v0_baseline` 是第一个完整 baseline，用于验证英文到中文机器翻译任务上的手写 Transformer Encoder-Decoder 训练闭环和推理闭环。

该版本已完成 48 epoch 训练，选择 valid loss 最低的 `checkpoint_epoch_20.pt` 作为 baseline checkpoint。完整训练日志、推理日志、固定测试集分析和错误分析已经提交到 `v0_baseline` 目录。

## v1_curriculum

`v1_curriculum` 是基于 `v0_baseline` 暴露的问题重新设计的课程学习版本，当前已补充 Stage1 到 Stage5 的训练记录与指标分析。

该版本不再只依赖单一 OPUS-100 语体，而是使用 Stage1 到 Stage5 的分阶段训练数据，让模型从基础短句逐步过渡到普通书面句、逻辑句、技术术语、复杂逻辑、mixed regression 和 anti-pollution stability。

v1 当前提交内容包括：

- Stage1 到 Stage5 的 `train.jsonl`
- 统一验证集 `final_eval.jsonl`
- 统一 48,000 vocab tokenizer
- v1 curriculum 训练脚本
- V1 final best checkpoint 分片文件
- v1 curriculum 推理脚本
- V1 主训练报告、V0/V1 对比报告、训练记录与指标分析
- V1 推理日志、推理报告和推理摘要

v1 当前不提交：

- stage 下的 `test.jsonl`
- 完整原始训练 raw log
- 中间阶段 checkpoint

当前推理报告基于最终 checkpoint 的 80 条分类样例输出生成，原始推理输出保留在 `encoder_decoder/v1_curriculum/logs/`。

## Dataset Work

v1 的课程数据来自本地整理后的多来源开源英中数据集合。数据构造阶段使用了 `available_data_unified/` 目录中的开源语料资源，包括通用平行语料、字幕语料、软件本地化语料、技术文档语料、教育类语料、机器翻译 benchmark 和术语/词典资源。

这些原始资源没有直接堆叠进训练集，而是经过筛选、清洗、格式统一、去重、语体划分和阶段化构造，最终形成 Stage1 到 Stage5 的课程学习数据。v1 的重点之一就是展示从数据收集到课程训练数据设计的完整数据工程能力。

## Model Configuration

`v0_baseline` 和 `v1_curriculum` 使用不同的实验配置。具体配置以各版本训练脚本为准。

v1 curriculum 当前训练脚本的主要配置：

| 配置 | 值 |
|------|----|
| architecture | Transformer Encoder-Decoder, Post-LN |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_heads` | 12 |
| `num_layers` | 10 encoder layers + 10 decoder layers |
| `dropout` | 0.08 |
| `max_src_len` | 96 |
| `max_tgt_len` | 96 |
| tokenizer vocab size | 48000 |
| validation set | `final_eval.jsonl` |

## Checkpoint Support

`v0_baseline` 和 `v1_curriculum` 的 checkpoint 策略不同。v0 主要用于保存可推理的 baseline 权重；v1 面向分阶段课程训练，因此训练脚本升级了完整续训能力。

### v0_baseline Checkpoints

v0 训练脚本的保存策略：

| 项目 | v0 行为 |
|------|---------|
| 周期保存 | 每 4 个 epoch 保存一次 `checkpoint_epoch_{epoch}.pt` |
| 最终保存 | 训练结束保存 `final_model.pt` |
| tokenizer | 保存训练时生成的 16,000 vocab tokenizer |
| checkpoint 内容 | `epoch` + `model_state_dict` |
| 自动续训 | 不支持完整自动续训 |
| optimizer / scheduler | 不保存 |

v0 当前提交的是 valid loss 最低的 baseline checkpoint：

```text
encoder_decoder/v0_baseline/checkpoints/checkpoint_epoch_20.pt
```

该 checkpoint 适合推理和作为权重分析对象，但无法精确恢复原训练现场。

### v1_curriculum Checkpoint Upgrade

v1 训练脚本面向 Stage1 到 Stage5 的连续课程训练。Stage1 从随机初始化开始；Stage2 到 Stage5 从上一个 stage 的 `best_checkpoint.pt` 继续训练。

v1 保存策略：

| 文件 | 触发时机 | 用途 |
|------|----------|------|
| `checkpoint_latest.pt` | 每个 epoch 后保存 | 中断恢复，保留最近训练现场 |
| `checkpoint_epoch_{epoch}.pt` | 每 4 个 epoch 保存 | 阶段内归档，便于回看特定 epoch |
| `best_checkpoint.pt` | 当前 run 的 valid loss 刷新最低值时保存 | 阶段传递 checkpoint，Stage2-Stage5 默认从上一阶段 best 开始 |

v1 checkpoint 内容升级为 `checkpoint_version=2`，包含：

| 字段 | 说明 |
|------|------|
| `epoch` | checkpoint 对应 epoch |
| `global_step` | 已完成的训练 step 数 |
| `is_final` | 是否为最终保存标记 |
| `model_state_dict` | 模型参数 |
| `optimizer_state_dict` | Adam optimizer 状态 |
| `scheduler_state_dict` | Noam / LambdaLR scheduler 状态 |
| `rng_state` | Python / Torch / CUDA RNG 状态 |
| `training_config` | 当前 stage、数据路径、模型结构、tokenizer、batch size、warmup、label smoothing 等配置 |
| `train_loss_history` | 当前 run 的 train loss 历史 |
| `valid_loss_history` | 当前 run 的 valid loss 历史 |
| `tokenizer_dir` | checkpoint 保存时配套 tokenizer 路径 |
| `pad_id`, `bos_id`, `eos_id` | 训练时使用的特殊 token id |

### Resume Behavior

v1 脚本恢复训练时会：

1. 加载 `model_state_dict`。
2. 如果存在 optimizer 状态，则恢复 optimizer 并移动到当前 CUDA device。
3. 如果存在 scheduler 状态，则恢复 scheduler。
4. 恢复 RNG state，减少中断恢复后的随机性漂移。
5. 恢复 `global_step`、train loss history 和 valid loss history。
6. 检查 checkpoint 中的关键配置是否与当前配置一致，包括 `vocab_size`、`d_model`、`d_ff`、`num_heads`、`num_layers`、`max_src_len` 和 `max_tgt_len`。

这使 v1 可以支持真正的阶段训练和中断续训，而不是只加载权重重新开始 optimizer。

### Stage Handoff Policy

v1 阶段传递规则：

```text
Stage1 random init
Stage1 best_checkpoint.pt -> Stage2
Stage2 best_checkpoint.pt -> Stage3
Stage3 best_checkpoint.pt -> Stage4
Stage4 best_checkpoint.pt -> Stage5
```

Stage5 结束后，最终 checkpoint 不只看 valid loss，还需要结合固定翻译样例表现综合选择。

### Current Repository Status

当前 GitHub 更新已提交 v1 训练数据、统一 tokenizer、训练脚本和训练分析文档。v1 checkpoint、推理报告和最终模型分析会在完成复核后再补充。

## How to Run

安装依赖：

```bash
pip install -r requirements.txt
```

运行 v0 baseline 训练：

```bash
bash scripts/run_train_encoder_decoder.sh
```

运行 v0 baseline 推理：

```bash
bash scripts/run_infer_encoder_decoder.sh
```

运行 v1 curriculum 训练：

```bash
bash scripts/run_train_encoder_decoder_v1_curriculum.sh
```

运行 v1 curriculum 推理：

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

## Notes for Reviewers

推荐阅读路径：

1. [v0_baseline README](./encoder_decoder/v0_baseline/README.md)
2. [v0 架构说明](./encoder_decoder/v0_baseline/doc/architecture_notes.md)
3. [v0 训练脚本](./encoder_decoder/v0_baseline/src/train_encoder_decoder.py)
4. [v0 推理报告](./encoder_decoder/v0_baseline/results/inference_report.md)
5. [v1_curriculum README](./encoder_decoder/v1_curriculum/README.md)
6. [v1 架构说明](./encoder_decoder/v1_curriculum/doc/architecture_notes.md)
7. [v1 数据说明](./encoder_decoder/v1_curriculum/data/README.md)
8. [v1 训练脚本](./encoder_decoder/v1_curriculum/src/train_encoder_decoder_curriculum.py)
9. [v1 主训练报告](./encoder_decoder/v1_curriculum/doc/v1_curriculum_training_report.md)
10. [v1 与 v0 对比报告](./encoder_decoder/v1_curriculum/doc/v1_vs_v0_comparison.md)
11. [v1 训练记录与指标分析](./encoder_decoder/v1_curriculum/doc/v1_training_metrics_summary.md)
12. [v1 checkpoint 说明](./encoder_decoder/v1_curriculum/checkpoints/README.md)
13. [v1 推理脚本](./encoder_decoder/v1_curriculum/src/infer_encoder_decoder_curriculum.py)
14. [v1 推理报告](./encoder_decoder/v1_curriculum/results/v1_inference_report.md)
15. [v1 推理日志摘要](./encoder_decoder/v1_curriculum/logs/v1_inference_log_summary.md)

后续 Encoder-Decoder 更新会根据真实训练进展补充到对应版本目录中，避免提前承诺尚未完成的训练结果。
