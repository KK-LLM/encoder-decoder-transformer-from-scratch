# V1 Curriculum Logs

本目录保存 V1 curriculum 推理相关日志。这里不放完整训练 raw log，只保留本轮最终 checkpoint 推理所需的输入、原始输出和结构化日志摘要。

## Files

| 文件 | 说明 |
|---|---|
| `v1_inference_cases.txt` | 本轮推理实际使用的 80 条英文输入句 |
| `v1_inference_raw_output.log` | 推理脚本的原始控制台输出，包含 80 条 EN/ZH 结果 |
| `v1_inference_log_summary.md` | 对 raw output 的结构化摘要和分类观察 |

## Inference Setup

本轮推理使用：

- checkpoint：`../checkpoints/checkpoint_best.pt`
- tokenizer：`../tokenizer/en_zh_stage_tokenizer_48000/`
- 推理脚本：`../src/infer_encoder_decoder_curriculum.py`
- batch size：16
- decode：batch greedy decode

样例覆盖 basic、general、general_logic、technical_terms、tech_logic、complex_logic、regression 和 anti_pollution 八类。

## Notes

`v1_inference_raw_output.log` 是原始输出，不做人工润色。报告中的分析基于该 raw log。
