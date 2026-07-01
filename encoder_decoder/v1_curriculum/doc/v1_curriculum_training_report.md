# V1 Curriculum 课程学习训练报告

## 总体说明

`v1_curriculum` 是我在 `v0_baseline` 基础上继续推进的课程学习训练版本。V0 主要解决的是从零实现 Encoder-Decoder Transformer 的完整闭环，包括 tokenizer、dataloader、training loop、checkpoint、推理脚本和固定样例评估；V1 则更关注训练数据怎么组织、训练难度怎么递进、不同阶段的能力怎么保留下来，以及在 loss 曲线出现问题时如何调整训练参数。

这次 V1 不是简单把数据量扩大，而是把训练过程拆成 Stage1 到 Stage5。每个阶段都有相对明确的训练目标：先打基础英中对齐，再扩展日常句和普通书面句，然后加入技术术语、逻辑结构、复杂句、mixed regression 和 anti-pollution 样本，最后做综合收束。

## 课程学习设计

| 阶段 | 训练集规模 | 阶段 test 文件规模（本次不作为验证集） | 统一验证集规模 | 日志覆盖 epoch | 最佳 epoch | 最低 valid loss | 最后一轮 valid loss | 备注 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Stage1 | 107,000 | 2,000 | 15,000 | 1-18 | 14 | 4.5573 | 4.6164 | valid loss 统一来自 `final_eval.jsonl` |
| Stage2 | 281,681 | 6,500 | 15,000 | 15-30 | 25 | 3.9273 | 3.9529 | valid loss 统一来自 `final_eval.jsonl` |
| Stage3 | 432,987 | 7,444 | 15,000 | 26-33 | 30 | 3.3192 | 3.3424 | valid loss 统一来自 `final_eval.jsonl` |
| Stage4 | 503,000 | 7,300 | 15,000 | 31-48 | 38 | 3.1282 | 3.1408 | valid loss 统一来自 `final_eval.jsonl` |
| Stage5 | 548,084 | 9,444 | 15,000 | 49-99 | 99 | 2.6998 | 2.6998 | valid loss 统一来自 `final_eval.jsonl`；低学习率版本连续训练并延长到 epoch99 |

## 统一训练设置

| 训练设置 | 当前配置 |
|---|---|
| 模型 | 手写 Transformer Encoder-Decoder，Post-LN |
| tokenizer | 统一 48,000 vocab tokenizer |
| 验证集 | 统一 `final_eval.jsonl`，15,000 条 |
| max length | source 96，target 96 |
| 训练设备 | NVIDIA GeForce RTX 5090 |
| 混合精度 | CUDA AMP，BF16 |
| checkpoint 策略 | 每个 epoch 保存 latest，每 4 个 epoch 保存归档，valid loss 刷新时保存 best |
| 固定样例类别 | basic、general_logic、technical_terms、complex_logic、regression_antipollution |

## 各阶段训练摘要

### Stage1：基础短句稳定化阶段

Stage1 从随机初始化开始，主要目标是让模型先学会最基本的英中对齐。这个阶段不追求复杂句，而是重点覆盖常见主语、动作、物体、时间、地点和简单日常句。

- 数据规模：train 107,000，统一 valid 15,000。
- 日志范围：epoch 1 到 epoch 18。
- 最佳 checkpoint：epoch 14，valid loss 4.5573。
- 最后可见 valid loss：4.6164。
- 观察结果：基础句已经开始形成可读输出。

### Stage2：普通日常句与简单逻辑扩展阶段

Stage2 从 Stage1 best checkpoint 继续训练，数据分布开始加入中等长度日常句、普通书面句和简单逻辑句。这个阶段的重点是让模型不只会翻短句，还能处理 because / if / when / but 这类基础逻辑结构。

- 数据规模：train 281,681，统一 valid 15,000。
- 日志范围：epoch 15 到 epoch 30。
- 最佳 checkpoint：epoch 25，valid loss 3.9273。
- 最后可见 valid loss：3.9529。
- 观察结果：valid loss 比 Stage1 明显下降，但后期仍出现 train loss 继续下降、valid loss 不再同步下降的情况，所以仍然需要按 best checkpoint 传递到下一阶段。

### Stage3：技术术语与技术语境增强阶段

Stage3 加入 ML / DL / Transformer / tokenizer / encoder / decoder / checkpoint / learning rate 等技术术语和技术句。这个阶段的关键不是做术语表，而是把术语放进自然句里训练，让模型在技术上下文中学习稳定翻译。

- 数据规模：train 432,987，统一 valid 15,000。
- 日志范围：epoch 26 到 epoch 33。
- 最佳 checkpoint：epoch 30，valid loss 3.3192。
- 最后可见 valid loss：3.3424。
- 观察结果：技术术语能力比 V0 明显改善，但仍会出现技术表达泛化、source 细节丢失和术语上下文不够稳的问题。

### Stage4：复杂逻辑与 mixed regression 防退化阶段

Stage4 把普通逻辑、复杂逻辑和技术逻辑混合起来，同时加入 general_written replay、easy replay 和 error_regression_mixed。这个阶段的目的不是单纯加长句子，而是让技术术语、逻辑结构和基础翻译能力尽量同时保留。

- 数据规模：train 503,000，统一 valid 15,000。
- 日志范围：epoch 31 到 epoch 48。
- 最佳 checkpoint：epoch 38，valid loss 3.1282。
- 最后可见 valid loss：3.1408。
- 观察结果：技术逻辑句有提升，但 complex_logic 固定样例仍能看到 source following 不稳定，说明复杂句结构还没有完全收住。

### Stage5：最终综合收束阶段

Stage5 是最终综合收束阶段，主要处理 easy replay、general_written replay、tech sentence replay、terminology stability、complex_logic、mixed_regression 和 anti-pollution。这个阶段不再是继续盲目加难度，而是尽量把基础句、技术句和复杂逻辑句之间的能力做平衡。

- 数据规模：train 548,084，统一 valid 15,000。
- 日志范围：epoch 49 到 epoch 99。
- 最佳 checkpoint：epoch 99，valid loss 2.6998。
- 最后可见 valid loss：2.6998。
- 观察结果：低学习率版本从 epoch49 到 epoch99 持续下降，但后期下降幅度已经很小，因此最终选择 epoch99 作为当前 V1 的 best epoch。

## Stage5 参数调整

Stage5 是最终收束阶段，我一开始没有把它当成一个普通的新阶段继续往下跑，而是重点观察 train loss 和 valid loss 是否还能同步改善。旧参数版本使用 `dropout=0.08`、`warmup_steps=7000`、`label_smoothing=0.03` 和 Noam `factor=1.0`。训练到后面时，我发现一个明显不正常的趋势：train loss 还在继续下降，但统一验证集 `final_eval.jsonl` 上的 valid loss 在短暂下降后开始反弹。

旧参数下，valid loss 从 epoch39 的 `3.0179` 降到 epoch40 的 `3.0089`，随后没有继续稳定下降，到 epoch52 已经回升到 `3.0771`；同时 train loss 从 epoch39 的 `3.3326` 降到 epoch52 的 `2.9358`。也就是说，旧参数下模型在训练集上还在继续下降，但验证集曲线已经开始变差，这说明继续沿用前面阶段的训练节奏并不适合 Stage5 的最终收束。

结合 loss 曲线和 Stage5 的训练目标，我主要把问题拆到以下几个训练参数上。

第一，学习率偏高。Stage5 是从前面阶段 checkpoint 继续训练，不是从随机初始化开始。如果还保持接近 `1e-4` 量级的学习率，参数更新步子会比较大，容易把 Stage4 已经学到的基础翻译、技术术语和逻辑结构扰动掉。我没有直接手写固定学习率，而是继续沿用 Noam scheduler，通过把 Noam `factor` 从 `1.0` 降到 `0.02` 来整体缩小学习率尺度。这样做的好处是保留原来的调度形式，同时把更新幅度降到更适合最终微调的范围。旧参数日志里的 LR 大约在 `9e-5` 到 `1.2e-4` 附近，新参数日志里的 LR 大约在 `1.1e-6` 到 `1.9e-6` 附近，更新明显更保守。

第二，`warmup_steps=7000` 对 Stage5 来说太长。新 Stage5 每个 epoch 约 5075 steps，7000 steps 已经超过一个 epoch。对于从 checkpoint 继续的最终收束阶段，我不希望模型在很长一段时间里还处在偏“重新启动训练”的 warmup 节奏中，而是希望它尽快进入稳定、低学习率的微调状态。所以我把 `warmup_steps` 调到 `1000`，让学习率调度更符合短周期收束，而不是继续沿用大阶段训练时的长 warmup。

第三，`label_smoothing=0.03` 对 Stage5 的目标不够合适。Stage5 里有大量术语稳定、source following 和 anti-pollution 样本，这些样本更强调目标 token 的精确性，比如 `tokenizer`、`encoder`、`decoder`、`checkpoint`、`special tokens` 这类词不应该被过度平滑成“差不多”的输出。较高的 label smoothing 会降低模型对正确 token 的置信度，在普通泛化训练中有帮助，但在最终收束阶段可能削弱术语和源句对齐的精确性。所以我把 `label_smoothing` 从 `0.03` 降到 `0.01`，保留一点平滑，但让模型更明确地贴近 target。

第四，`dropout=0.08` 在最终阶段可能偏强。Stage5 不是重新学习一个大分布，而是要在已有 checkpoint 上收紧输出稳定性。过强 dropout 会给每一步训练带来更多随机扰动，容易让术语稳定、复杂逻辑和 anti-pollution 这些精细能力不够收敛。因此我把 `dropout` 降到 `0.05`，保留基本正则化，同时减少最终阶段的训练噪声。

基于这些分析，我只改了四个参数：`dropout` 从 `0.08` 降到 `0.05`，`warmup_steps` 从 `7000` 降到 `1000`，`label_smoothing` 从 `0.03` 降到 `0.01`，Noam `factor` 从 `1.0` 降到 `0.02`。这里没有改模型结构，也没有重新构造 Stage5 数据，目的是尽量把变量集中在最终收束参数上。

新参数训练后，走势明显比旧参数健康。旧参数虽然 train loss 从 `3.3326` 降到 `2.9358`，但 valid loss 从最低 `3.0089` 回升到 `3.0771`，属于训练集继续下降、验证集反弹的状态。新参数一开始的 valid loss 就降到 `2.7900`，后面继续下降到 epoch99 的 `2.6998`；train loss 也从 `3.1373` 降到 `2.8501`。这说明低学习率、更短 warmup、更低 label smoothing 和更低 dropout 更适合 Stage5 的最终收束。后期下降幅度已经明显变小，所以我选择在 epoch99 停止，并把它作为当前 V1 的 final best epoch。

## 固定样例观察

固定样例不能替代完整评估，但它能帮助我观察 valid loss 看不到的细节。比如基础句有没有漏掉动作和时间，技术术语有没有漂移，逻辑连接词有没有保留，anti-pollution 样例会不会触发重复输出或模板化表达。

V1 的固定样例整体能看到几类变化：基础日常句比 V0 更稳定，技术术语更容易被保留，`encoder`、`decoder`、`tokenizer`、`checkpoint`、`learning rate` 等词不再像 V0 那样大面积失真；逻辑句也比 baseline 更可读。但 complex_logic 仍然不稳，长句里偶尔会丢从句关系，技术句里也还会出现中英文混排、token 空格和表达模板化问题。

## Final best epoch 样例快照

| 样例类别 | 英文输入（固定样例） | 模型输出 |
|---|---|---|
| `basic` | the girl opened the window yesterday. | 女 孩 昨 天 打 开 了 这 扇 窗 户 。 |
| `general_logic` | Although some progress has been made, much remains to be done. | 尽 管 取 得 了 进 展 ， 但 仍 有 许 多 工 作 要 做 。 |
| `technical_terms` | The encoder output gives the decoder source-side context in sequence-to-sequence translation. | 编 码 器 输 出 会 为 解 码 器 提 供 源 端 上 下 文 ， 适 用 于 序 列 到 序 列 翻 译 。 |
| `complex_logic` | Because there was not enough time, the child could not wait in the room. | 因 为 时 间 不 够 ， 那 个 孩 子 无 法 在 房 间 里 等 。 |
| `regression_antipollution` | If the tokenizer produces unstable token sequences, translation quality may drop. In LLM inference, engineers use special tokens to avoid fragmented subwords. | 如 果 分 词 器 生 成 的 token 序 列 不 稳 定 ， 翻 译 质 量 可 能 会 下 降 。 在 语 言 模 型 推 理 中 ， 工 程 师 使 用 特 殊 token 来 避 免 碎 片 化 子 词 |

## 当前训练结果总结

V1 相比 V0，最主要的变化是训练路线更清楚。V0 重点是把手写 Encoder-Decoder Transformer 的基础训练闭环跑通；V1 则把能力拆成几个阶段，让模型先学习基础英中对齐，再扩展到普通日常句、技术术语、逻辑句、复杂逻辑和 anti-pollution 场景。

从统一 valid loss 看，Stage1 best 为 `4.5573`，Stage2 best 为 `3.9273`，Stage3 best 为 `3.3192`，Stage4 best 为 `3.1282`，Stage5 best 为 `2.6998`。因为这些阶段都使用同一个 `final_eval.jsonl`，所以这条曲线可以作为跨阶段训练走势的参考。

从固定样例看，基础句、技术术语和一部分逻辑句的输出比 V0 更稳定。尤其是 `tokenizer`、`encoder`、`decoder`、`checkpoint`、`Transformer` 这类技术词，在 final best epoch 中已经能更稳定地保留或翻译。basic 样例也能比较稳定地保留主语、动作、时间和宾语。

这版 V1 对我来说还有一个重要收获：训练过程不只是准备数据和等待 epoch 结束。Stage5 旧参数曲线出现问题后，我根据 train loss / valid loss 的分化去分析学习率、warmup、label smoothing 和 dropout，再用新参数重新训练并观察曲线变化。这个过程把数据构造、训练诊断、参数调整和 checkpoint 选择串了起来。

## 当前不足

当前主要问题包括：

- `complex_logic` 仍然不稳定。对应到本次固定样例和阶段日志，复杂句里虽然有时能保留 because / although 这类连接词，但长句和多从句仍容易出现漏译、关系弱化或语义转移。
- 长句 source following 仍有偏移。模型有时能抓住主题词，但对英文源句里的限定条件、先后关系和从句边界跟随不够稳定，技术长句中尤其容易压缩或改写细节。
- 当前模型仍是 Post-LN 结构。Stage5 旧参数下 valid loss 容易反弹，后面需要把学习率大幅降下来才更稳定，这说明当前深层 Encoder-Decoder 对学习率和正则化仍比较敏感。
- 当前归一化层仍使用 LayerNorm。训练后期对 dropout、label smoothing 和学习率变化比较敏感，说明整体归一化和数值稳定性还有继续优化空间。
- 当前 Encoder / Decoder stack 末尾没有单独的 final norm。对应到生成结果上，部分技术句和 anti-pollution 样例已经能翻出主体意思，但尾部输出、标点和 token 边界仍不够稳定，例如特殊 token、subwords 相关句子容易出现形式不够规整的问题。
- 当前 FFN 仍是 ReLU FFN。对于复杂逻辑和技术长句，模型有时能翻出关键词，但对多条件关系和术语上下文的组合表达还比较朴素，说明中间层的表达能力还可以继续加强。
- 当前输出层和 target embedding 没有做更细的参数共享设计。对应到词表侧表现，技术术语、英文 token 和中文子词之间的输出形式还不够统一，后续还有提升参数效率和词表表示一致性的空间。
- 当前推理主要基于 greedy decode。对于长句、复杂句和术语句，greedy decode 的候选搜索不够充分，容易过早选择局部最优 token，导致后续句子结构或细节表达变弱。

## 后续计划

后续我会继续沿着两个方向推进。第一是训练结构本身的改进，包括归一化、FFN、输出层和解码策略这些会影响稳定性和生成质量的部分。第二是保持小步实验和对照记录，先确认每一类结构调整对训练曲线和固定样例的影响，再决定是否合入下一版主线。

V2 不会只是在 V1 后面继续堆数据，而是会结合这次 V1 暴露出来的问题，对模型结构和推理策略做更系统的更新。
