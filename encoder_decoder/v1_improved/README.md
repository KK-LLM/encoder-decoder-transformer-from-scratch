# Encoder-Decoder v1_improved

计划中的优化版本，将在 v0_baseline 基础上改进训练策略和模型配置。

## 计划内容

- 优化数据配比和数据清洗策略
- 调整模型规模（d_model / num_layers / num_heads）
- 优化 learning rate schedule
- 添加 beam search 解码
- 支持 checkpoint 自动续训
- 抽离 model.py 模块
- 添加 BLEU / chrF 评估

## 状态

计划中，预计在 v0_baseline 训练完成并分析结果后启动。
