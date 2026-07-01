# Encoder-Decoder Transformer

该目录只整理 Encoder-Decoder Transformer from scratch 相关版本。当前仓库不包含 Decoder-only from scratch。Decoder-only 后续会单独新建仓库，不属于当前 Encoder-Decoder 仓库。

## Versions

| 版本 | 目录 | 状态 | 说明 |
|------|------|------|------|
| v0_baseline | [v0_baseline/](./v0_baseline/) | 训练已完成 | 基于 OPUS-100 en-zh 的 baseline 版本，已提交 checkpoint、训练日志、推理日志和错误分析 |
| v1_curriculum | [v1_curriculum/](./v1_curriculum/) | 已补充训练记录 | 基于 Stage1 到 Stage5 的课程学习版本，已提交训练数据、48k tokenizer、训练脚本和训练分析文档 |

## Version Relationship

`v0_baseline` 用于验证手写 Encoder-Decoder Transformer 的基本训练闭环和推理闭环。该版本暴露出数据分布单一、技术术语不稳定、日常英语泛化不足、复杂逻辑句处理较弱等问题。

`v1_curriculum` 是在这些问题基础上重新设计的数据与训练方案。它使用统一 48,000 vocab tokenizer 和统一验证集 `final_eval.jsonl`，并按 Stage1 到 Stage5 逐步训练：

1. Stage1：基础短句和日常表达
2. Stage2：普通日常句、普通书面句和简单逻辑
3. Stage3：技术句和术语稳定
4. Stage4：综合逻辑、复杂逻辑和技术逻辑
5. Stage5：最终收束、mixed regression、防遗忘和 anti-pollution stability

v1 当前已补充 Stage1 到 Stage5 的训练记录、V0/V1 对比和指标分析。推理报告、checkpoint 和最终模型分析会在后续完成复核后再补充。
