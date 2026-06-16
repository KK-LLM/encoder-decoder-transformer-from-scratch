# 项目路线图

## 已完成

- [x] Encoder-Decoder v0_baseline：手写 Transformer + 训练闭环 + 推理闭环 + tokenizer + opus-100 数据
- [x] 项目结构整理、文档完善

## v0_baseline 后续（训练完成后）

- [ ] 补充完整训练日志摘要
- [ ] 添加 BLEU / chrF 评估指标
- [ ] 添加典型推理样例和错误分析
- [ ] 上传最终 checkpoint（Git LFS）

## v1_improved（计划中）

- [ ] 优化数据配比和数据清洗策略
- [ ] 调整模型规模（d_model / num_layers / num_heads）
- [ ] 优化 learning rate schedule 和 warmup 策略
- [ ] 添加 beam search 解码
- [ ] 支持 checkpoint 自动续训
- [ ] 引入 config.yaml 配置系统
- [ ] 抽离 model.py 模块
- [ ] 引入验证集 BLEU early stopping

## Decoder-only（计划中）

- [ ] 手写 GPT-style Decoder-only 架构
- [ ] 训练闭环（语言建模 / next-token prediction）
- [ ] 推理闭环（自回归文本生成）
- [ ] 支持 KV Cache 推理加速

## 长期展望

- [ ] LoRA / QLoRA 微调实践
- [ ] 从零训练 vs 微调预训练模型的对比实验
- [ ] 模型蒸馏和量化实验

> 注：LoRA 微调、数据蒸馏、业务模型优化等偏应用的项目，建议另开独立仓库，不与本底层手写模型仓库混合。
