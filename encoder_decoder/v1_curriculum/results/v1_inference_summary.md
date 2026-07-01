# V1 Curriculum 推理摘要

这次推理使用最终 checkpoint `./checkpoint_best.pt`，配合 `./Encoder-Decoder-中英文翻译数据-推理-0630.py` 对 80 条英文样例做 batch greedy decode。样例按 basic、general、general_logic、technical_terms、tech_logic、complex_logic、regression 和 anti_pollution 八类组织，用来观察 V1 final best epoch 的真实输出状态。

## 推理记录

| 推理内容 | 实际使用 |
|---|---|
| checkpoint | `./checkpoint_best.pt` |
| epoch | 99 |
| global_step | 394463 |
| tokenizer | `./en_zh_stage_tokenizer_48000` |
| vocab size | 48000 |
| 推理设备 | Apple Silicon MPS |
| batch size | 16 |
| 原始输出 | `./inference_logs/v1_inference_raw_output.log` |

## 主要结论

- basic 类有一部分已经比较稳定，例如 `the girl opened the window yesterday.` 输出为“女 孩 昨 天 打 开 了 这 扇 窗 户 。”。
- technical_terms 是 V1 相比 V0 提升最明显的部分。`tokenizer`、`checkpoint`、`learning rate`、`encoder`、`decoder`、`Transformer` 等术语在多条样例中能被保留或正确翻译。
- tech_logic 有部分可用输出，例如 `If the tokenizer produces unstable token sequences, translation quality may drop.` 输出为“如 果 分 词 器 不 稳 定 ， 翻 译 质 量 可 能 会 下 降 。”。
- general 和 complex_logic 仍然不稳，模型经常生成流畅但偏离 source 的中文。
- regression 样例显示 replay 和 mixed regression 有一定效果，但 book/notebook/laptop、door/window、open/close 仍会混淆。
- anti_pollution 没有出现严重无限重复，也没有直接输出固定规则前缀，但仍然会出现说明式或训练报告式表达。

## 当前判断

V1 的推理表现和训练报告中的判断是一致的：课程学习确实扩展了模型能力，尤其是基础短句和技术术语；但复杂逻辑、普通书面句、source following 和 anti-pollution 还没有完全收稳。

这批推理结果可以作为 V1 当前阶段的真实记录。后续整理到 GitHub 时，建议保留 raw output，同时把推理报告、推理摘要和典型错误样例放到 `encoder_decoder/v1_curriculum/results/` 或对应文档目录中。
