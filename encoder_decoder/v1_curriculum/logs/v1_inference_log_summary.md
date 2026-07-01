# V1 Curriculum 推理日志摘要

本文件是 `v1_inference_raw_output.log` 的结构化摘要，用来记录这次推理实际跑了什么、输出是否完整，以及从原始日志中能看到哪些直接结果。原始控制台输出不做改写，仍保存在 `v1_inference_raw_output.log`。

## 运行记录

| 推理内容 | 实际使用 |
|---|---|
| checkpoint | `./checkpoint_best.pt` |
| checkpoint epoch | 99 |
| global_step | 394463 |
| tokenizer | `./en_zh_stage_tokenizer_48000` |
| vocab size | 48000 |
| 模型配置 | `d_model=768`，`d_ff=3072`，`heads=12`，`layers=10`，`dropout=0.05` |
| max_src_len | 96 |
| 推理设备 | Apple Silicon MPS |
| batch size | 16 |
| 输入样例 | 80 条 |
| 原始输出 | `./inference_logs/v1_inference_raw_output.log` |

## 输出完整性

原始日志中共有 80 条 `EN` 和 80 条 `ZH` 输出，编号从 `[1]` 到 `[80]` 连续，没有发现中断、报错或缺失编号。

## 分类观察

| 类别 | 日志中的主要现象 |
|---|---|
| `basic` | 一部分基础短句能正确输出，例如 window、door、notebook、desk 相关句子；仍有 book/desk、coffee/tea、chair/table 等错配。 |
| `general` | 输出能读，但 source following 较弱，容易把普通书面句改写成泛化中文。 |
| `general_logic` | 如果、因为、当、虽然等连接词更容易出现，但连接词两边的语义仍会漂移。 |
| `technical_terms` | tokenizer、checkpoint、learning rate、encoder、decoder、Transformer 等术语明显比 V0 稳定。 |
| `tech_logic` | 部分技术逻辑句可用，但多个术语和多个动作同时出现时仍会模板化。 |
| `complex_logic` | 多从句和复杂逻辑仍然是弱项，常见问题是漏译、改写和泛泛中文输出。 |
| `regression` | 部分 book/notebook/door/window 样例正确，但短祈使句和 book/laptop 组合仍不稳。 |
| `anti_pollution` | 没有严重无限重复和固定规则前缀，但仍会出现解释式或训练报告式输出。 |

## 典型输出摘录

| 编号 | 类别 | EN | ZH | 观察 |
|---:|---|---|---|---|
| 1 | basic | the girl opened the window yesterday. | 女 孩 昨 天 打 开 了 这 扇 窗 户 。 | 基础短句基本正确 |
| 32 | technical_terms | The tokenizer converts raw text into token ids before training starts. | 在 模 型 读 取 文 本 之 前 ， 分 词 器 会 将 原 始 文 本 转 换 为 token id 。 | tokenizer 术语和核心语义保留 |
| 41 | tech_logic | If the tokenizer produces unstable token sequences, translation quality may drop. | 如 果 分 词 器 不 稳 定 ， 翻 译 质 量 可 能 会 下 降 。 | 技术术语和 if 结构同时保留 |
| 57 | complex_logic | When users ask complex questions, the system should preserve the original meaning instead of generating generic answers. | 解 决 方 案 是 一 个 简 单 的 问 题 ， 因 为 它 可 以 使 用 一 个 简 单 的 方 法 来 解 决 问 题 。 | source following 明显失败 |
| 73 | anti_pollution | The tokenizer tokenizer tokenizer sequence should not make the output repeat the same word forever. | 分 词 器 不 是 字 典 ， 而 是 为 了 把 句 子 切 分 成 token 。 | 没有无限重复，但输出偏模板化 |

## 当前结论

从这次日志看，V1 final best epoch 的推理结果和训练阶段判断基本一致：基础句和技术术语有明显提升，复杂逻辑、普通书面句和 anti-pollution 仍然没有完全稳定。后续整理到 GitHub 时，可以把 raw output 作为原始证据，把本摘要和推理报告作为分析文档。
