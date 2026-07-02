# V1 Curriculum Results

本目录保存 V1 curriculum 当前最终 checkpoint 的推理报告和推理摘要。

## Files

| 文件 | 说明 |
|---|---|
| `v1_inference_report.md` | V1 完整推理报告，包含分类样例、V0/V1 对比和错误分析 |
| `v1_inference_summary.md` | V1 推理摘要，适合 README 或文档索引引用 |

## Current Inference Conclusion

本轮推理结果和训练报告中的判断基本一致：V1 相比 V0 的主要提升体现在基础短句和技术术语上，`tokenizer`、`checkpoint`、`learning rate`、`encoder`、`decoder`、`Transformer` 等术语已经比 V0 稳定。

但 V1 仍然存在明显不足，尤其是普通书面句、复杂逻辑句、source following 和 anti-pollution 场景。相关结论以 `v1_inference_report.md` 为准。
