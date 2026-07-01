# V1 Curriculum 与 V0 Baseline 对比报告

## 总体对比

| 对比维度 | V0 Baseline | V1 Curriculum |
|---|---|---|
| 训练目标 | 验证手写 Encoder-Decoder Transformer 的训练和推理闭环。 | 通过课程学习控制数据难度，让模型能力按阶段递进。 |
| 数据组织 | 主要使用 OPUS-100 en-zh 子集，单一 baseline 分布。 | Stage1 到 Stage5 分阶段构造，包含 replay、技术术语、逻辑句、复杂逻辑、mixed regression 和 anti-pollution。 |
| 训练规模 | V0 训练实际使用约 500,000 条。 | 107,000 -> 281,681 -> 432,987 -> 503,000 -> 548,084。 |
| 验证方式 | OPUS validation 2,000 条。 | 统一 `final_eval.jsonl`，15,000 条，跨阶段可比较。 |
| tokenizer | OPUS 数据上训练的 16,000 vocab tokenizer。 | Stage1-Stage5 数据上构造的统一 48,000 vocab tokenizer。 |
| 模型配置 | d_model=512，d_ff=2048，heads=8，layers=8，dropout=0.1。 | d_model=768，d_ff=3072，heads=12，layers=10；Stage1-4 dropout=0.08，Stage5 新参数 dropout=0.05。 |
| checkpoint 机制 | 主要保存 `epoch` 和 `model_state_dict`，不支持完整恢复 optimizer/scheduler。 | 保存内容更完整：除模型权重外，还保存 optimizer、scheduler、RNG、config、loss history、global_step 和 tokenizer 信息，更适合中断恢复和阶段传递。 |
| 最佳 valid loss | epoch 20，valid loss 3.4693。 | Stage5 epoch 99，valid loss 2.6998。 |
| 固定样例 | V0 能看到基础短句效果，但技术术语和逻辑句大量失败。 | V1 按 basic、general_logic、technical_terms、complex_logic、regression_antipollution 分桶观察。 |
| 训练诊断 | V0 识别出 epoch20 后过拟合。 | V1 根据 Stage5 的 train/valid 分化重新调整最终收束参数。 |

## V1 的主要提升

V1 的提升不只是训练轮次更多，也不只是数据规模变大。更关键的变化是：V1 开始围绕不同翻译能力设计数据分布，并且通过 Stage1 到 Stage5 的顺序训练，让这些能力逐步叠加。

1. 基础短句翻译更稳定。

V0 已经能看到一些基础短句的翻译能力，但输出还比较容易受数据分布影响。V1 从 Stage1 开始专门训练 easy_basic、basic_daily_sentence、simple_action、simple_object、simple_time、simple_place、simple_person 等样本，使模型先建立比较稳定的主语、动作、宾语、时间和地点映射。到 final best epoch，类似 `the girl opened the window yesterday.` 这类句子已经能比较稳定地保留主语、动作、时间和宾语。

2. 普通日常句和书面句覆盖更完整。

V0 更像是一个 baseline 翻译闭环，普通句子的覆盖主要依赖原始数据分布。V1 在 Stage2 之后加入 medium_daily_sentence、general_written、basic_general_translation 等数据，让模型从短句扩展到中等长度日常句和普通书面表达。这部分提升让模型不只会翻模板化短句，也能处理更自然的生活、学习、工作类表达。

3. 简单逻辑句有明显改善。

V0 对 because、if、when、although、but 这类逻辑连接词的保持能力较弱，容易把逻辑关系翻散。V1 从 Stage2 开始加入 simple_logic_sentence，Stage4 和 Stage5 继续加入 general_logic、complex_logic 和 mixed_regression。最终样例中，`Although some progress has been made, much remains to be done.` 能输出为“尽管取得了进展，但仍有许多工作要做”，说明让步关系和转折结构已经比 V0 更稳定。

4. 技术术语翻译是 V1 最明显的突破点之一。

V0 中技术句经常出现大幅失真，`Transformer`、`tokenizer`、`attention`、`checkpoint` 这类词不能稳定保留。V1 在 Stage3 专门加入 tech_sentence、terminology、real_tech_sentence、simple_logic_with_tech、terminology_stability 等数据，把技术术语放在自然句里训练。到 Stage5，`encoder`、`decoder`、`tokenizer`、`checkpoint`、`learning rate`、`Transformer` 等术语已经能更稳定地保留或翻译。

5. 技术语境下的 source following 更好。

V0 在技术句里容易生成泛化中文解释，而不是跟随 source。V1 加入技术句、技术逻辑句和 anti-pollution 样本后，模型对源句结构的跟随能力有所改善。例如 `The encoder output gives the decoder source-side context in sequence-to-sequence translation.` 能翻出“编码器输出”“解码器”“源端上下文”“序列到序列翻译”等关键信息，说明技术句已经不再只是随机保留术语，而是能保留一部分句内关系。

6. 复杂逻辑和 mixed regression 开始进入训练目标。

V0 对复杂逻辑基本没有专项训练。V1 在 Stage4 和 Stage5 中加入 complex_logic、tech_logic_sentence、mixed_regression、error_regression_mixed 和 replay 样本，让模型开始处理多从句、技术逻辑混合句和防遗忘问题。虽然 complex_logic 仍然没有完全解决，但它已经从 V0 的明显短板，变成了 V1 中可以被持续观察和优化的能力项。

7. anti-pollution 和术语稳定性有单独考虑。

V0 更关注能不能训练和推理，较少单独处理输出污染问题。V1 在 Stage5 中加入 anti-pollution stability 和 terminology_stability，希望减少规则说明句、模板句、重复 token、术语误译和泛化技术解释。final best epoch 中，anti-pollution 样例已经能较好保留 `tokenizer`、`token sequences`、`special tokens`、`fragmented subwords` 等核心信息。

8. checkpoint 和训练诊断更完整。

V0 已经能保存 checkpoint，但更偏向基础训练记录。V1 的 checkpoint 保存内容更完整，也更适合阶段传递和中断恢复。Stage5 中旧参数出现 train loss 下降但 valid loss 反弹后，我重新分析学习率、warmup、label smoothing 和 dropout，并调整到更适合最终收束的参数。这部分让 V1 不只是“训练完”，而是形成了更完整的训练诊断和调参闭环。

## 仍未完全解决的问题

- complex_logic 仍然是最明显的弱项。V1 已经开始加入 complex_logic 和 tech_logic_sentence，但长句、多从句、让步关系、因果关系叠加时，模型仍可能出现漏译、关系弱化或语义转移。
- 长句 source following 还不够稳定。技术长句中，模型有时能保留关键词，但会压缩源句细节，或者把英文中的限定条件翻得不够完整。
- 技术术语虽然比 V0 稳定，但术语上下文仍有波动。有些句子可以正确保留 `tokenizer`、`encoder`、`decoder`，但涉及多个术语和多个动作时，模型仍可能把关系翻得比较松。
- anti-pollution 还没有完全解决。V1 已经减少了一部分泛化技术解释和重复表达，但在更长的技术句或多句组合里，仍可能出现模板化中文、重复结构或过度概括。
- 输出形式仍有改进空间。当前结果中仍能看到中文 token 之间有空格、英文 token 和中文混排不够自然等问题，这会影响最终可读性。
- 推理策略仍比较基础。当前主要依赖 greedy decode，对长句、复杂句和术语句的候选搜索不够充分，容易在前几个 token 做出局部选择后影响后续句子结构。
- 模型结构仍是相对基础的 Encoder-Decoder Transformer。Post-LN、LayerNorm、ReLU FFN、无 final norm、输出层和 embedding 未做更细参数共享等设计，都让后续结构优化还有空间。

## V0 经验如何进入 V1

V0 证明了手写模型可以完成训练闭环，也暴露出几个关键问题：技术术语崩溃、泛化中文幻觉、逻辑结构弱、epoch20 后过拟合。V1 把这些问题转成了具体训练设计：技术术语数据、逻辑句数据、replay、anti-pollution 样例、统一验证集、固定样例分桶和更完整的 checkpoint 保存。
