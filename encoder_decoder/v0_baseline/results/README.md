# Results

v0_baseline 训练和推理阶段的定性分析报告。

## Files

| 文件 | 说明 |
|---|---|
| `translation_examples.md` | 训练过程中 5 个代表性 epoch 的固定例句翻译对比，展示 model 从欠拟合 → best epoch → 过拟合的翻译行为变化 |
| `inference_report.md` | epoch 20 checkpoint 推理报告，包含 fixed 三类专项测试、定性示例、错误分析和泛化分析 |
| `inference_examples.md` | epoch 20 推理精选样例，按 good / partial / poor 人工分类，覆盖 fixed_* 和 train/validation/test |
