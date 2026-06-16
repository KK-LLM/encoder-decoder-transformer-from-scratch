# Encoder-Decoder Transformer from Scratch

> Transformer / LLM from scratch learning and training lab.

从零手写 Transformer 模型结构、训练闭环、推理闭环的实验仓库。当前版本为 Encoder-Decoder v0_baseline，后续将逐步扩展优化版本。

## Why This Repo

- 从零实现 Transformer 核心模块（MultiHeadAttention、PositionalEncoding、LayerNorm 等），不依赖 `nn.Transformer`
- 完整训练闭环：数据加载 → tokenizer 训练 → 前向传播 → loss 计算 → Noam LR → checkpoint 保存
- 完整推理闭环：tokenizer 加载 → checkpoint 加载 → greedy decode → 翻译输出
- 每个版本保持独立目录，清晰展示迭代过程

## Current Progress

| 项目 | 版本 | 架构 | 状态 |
|------|------|------|------|
| Encoder-Decoder | [v0_baseline](./encoder_decoder/v0_baseline/) | Post-LN Transformer | 训练进行中 |
| Encoder-Decoder | v1_improved | 待定 | 计划中 |

## Repository Structure

```
.
├── README.md                   # 仓库总览
├── LICENSE
├── requirements.txt
├── .gitignore
├── .gitattributes
├── docs/
│   ├── roadmap.md              # 路线图
│   ├── architecture_notes.md   # 模型架构说明
│   ├── experiment_notes.md     # 实验记录模板
│   └── checkpoint_and_reproducibility.md
├── encoder_decoder/
│   ├── README.md
│   ├── v0_baseline/            # ← 当前版本
│   │   ├── README.md
│   │   ├── src/
│   │   │   ├── train_encoder_decoder_opus.py
│   │   │   └── infer_encoder_decoder.py
│   │   ├── data/
│   │   │   ├── README.md
│   │   │   ├── sample_data.jsonl         # 200 条数据样例
│   │   │   └── opus100_en_zh_local/      # 完整数据集 (Git LFS)
│   │   ├── tokenizer/
│   │   │   ├── README.md
│   │   │   ├── tokenizer.json
│   │   │   └── tokenizer_config.json
│   │   └── checkpoints/
│   │       └── README.md
│   └── v1_improved/            # 计划中
│       └── README.md
└── scripts/
    ├── run_train.sh
    └── run_infer.sh
```

## Notes for Reviewers

推荐按以下顺序阅读，快速了解项目：

1. **本文 (README.md)** — 仓库总览和当前进度
2. **[v0_baseline README](./encoder_decoder/v0_baseline/README.md)** — 当前版本的训练配置、数据说明、当前状态
3. **[train_encoder_decoder_opus.py](./encoder_decoder/v0_baseline/src/train_encoder_decoder_opus.py)** — 手写模型源码 + 训练闭环
4. **[infer_encoder_decoder.py](./encoder_decoder/v0_baseline/src/infer_encoder_decoder.py)** — 推理闭环
5. **[architecture_notes.md](./docs/architecture_notes.md)** — 模型架构说明
6. **[roadmap.md](./docs/roadmap.md)** — 长期计划和后续版本规划
