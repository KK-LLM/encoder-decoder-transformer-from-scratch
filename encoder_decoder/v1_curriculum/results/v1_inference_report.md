# V1 Curriculum 推理报告

## 总体说明

这次推理是基于当前 V1 最终 checkpoint 做的一轮固定样例观察。我没有重新训练模型，也没有修改 checkpoint 和模型结构，只是用最终权重跑一组覆盖 Stage1 到 Stage5 能力范围的英文句子，观察模型在真实推理时到底能稳定处理哪些内容，哪些问题仍然明显。

这份报告和训练报告的定位一致：不只看单个好例子，也不把 valid loss 或少数样例当成完整结论。我更关注的是基础句、技术术语、逻辑关系、复杂句、回归样例和 anti-pollution 样例在最终 checkpoint 上的实际表现。

## 推理设置

| 推理内容 | 实际使用 |
|---|---|
| checkpoint | `./checkpoint_best.pt` |
| checkpoint epoch | 99 |
| global_step | 394463 |
| 推理脚本 | `./Encoder-Decoder-中英文翻译数据-推理-0630.py` |
| tokenizer | `./en_zh_stage_tokenizer_48000` |
| vocab size | 48000 |
| 输入文件 | `./inference_cases/v1_inference_cases.txt` |
| 输入备份 | `./inference_logs/v1_inference_cases.txt` |
| 原始推理输出 | `./inference_logs/v1_inference_raw_output.log` |
| 推理设备 | Apple Silicon MPS |
| batch size | 16 |
| 解码方式 | batch greedy decode |
| 样例数量 | 80 |

推理脚本当前只暴露 `--input-file` 和 `--batch-size` 两个参数。为了不改动脚本文件，我在运行时只覆盖默认的 `MODEL_PATH` 和 `TOKENIZER_DIR`，让脚本使用当前目录下的 `checkpoint_best.pt` 和 `en_zh_stage_tokenizer_48000`。这样可以保持推理脚本和 checkpoint 本身不变，同时完成最终模型的批量推理。

checkpoint 中读取到的模型配置如下：

| 模型配置 | checkpoint 读取结果 |
|---|---:|
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_heads` | 12 |
| `num_layers` | 10 |
| `dropout` | 0.05 |
| `max_src_len` | 96 |

## 推理样例设计

这次一共构造了 80 条英文句子，每类 10 条。分类方式和 V1 训练报告里的能力观察口径保持一致。

| 类别 | 观察重点 |
|---|---|
| `basic` | 基础短句、日常动作、人物、物体、时间地点 |
| `general` | 普通书面句和中等长度句子 |
| `general_logic` | because / if / when / although / however 等逻辑关系 |
| `technical_terms` | Transformer、tokenizer、checkpoint、learning rate、encoder、decoder、attention、embedding 等术语 |
| `tech_logic` | 技术术语和条件、因果、转折结构的组合 |
| `complex_logic` | 多从句、让步、因果、条件组合句 |
| `regression` | door/window、book/notebook、today/tomorrow、open/close 等历史易错点 |
| `anti_pollution` | 规则说明句、模板句、泛泛技术中文和重复短语倾向 |

## 整体结果

这轮推理结果和训练阶段的观察基本一致：V1 相比 V0 的提升主要体现在基础短句和技术术语上。V0 baseline 在技术句上经常直接失效，例如把 `Transformer models use attention mechanisms.` 翻成“技 术 准 则 的 实 施 情 况 。”，把 `The training loss is decreasing.` 翻成“这 是 一 个 新 的 失 业 控 制 系 统 。”。V1 这次至少已经能稳定保留一批技术词，比如 `tokenizer`、`checkpoint`、`learning rate`、`encoder`、`decoder` 和 `Transformer`。

但这并不代表 V1 的翻译质量已经全面稳定。普通书面句和复杂逻辑句仍然有明显的 source following 问题。模型有时会生成流畅中文，但内容并没有严格跟随英文源句；技术句也会触发训练报告式或模板式输出。也就是说，V1 的能力覆盖范围比 V0 大了很多，但复杂句、长句和跨概念组合句仍然是当前短板。

## 分类推理结果

### basic：基础短句

basic 是这次推理里相对最稳的一类。模型已经能处理一部分主语、动作、宾语和时间比较清楚的短句。

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 1 | the girl opened the window yesterday. | 女 孩 昨 天 打 开 了 这 扇 窗 户 。 | 基本正确，主语、时间、动作和宾语都保留 |
| 2 | please close the door before dinner. | 请 晚 饭 前 关 上 这 扇 门 。 | 基本正确 |
| 3 | my brother put the notebook on the desk this morning. | 我 哥 哥 今 天 早 上 把 这 本 笔 记 本 放 在 书 桌 上 。 | 基本正确，notebook 和 desk 都保留 |

同时，basic 类仍然能看到基础错配：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 4 | the teacher will bring the book to the classroom tomorrow. | 老 师 明 天 在 教 室 里 会 带 来 这 张 书 桌 。 | book 被错译成书桌 |
| 5 | she drinks a cup of coffee in the kitchen every morning. | 她 每 天 早 上 在 厨 房 里 喝 一 杯 茶 。 | coffee 被错译成茶 |
| 8 | the man moved the chair near the table after lunch. | 那 位 男 士 午 饭 后 在 桌 子 旁 打 开 了 这 张 桌 子 。 | moved chair 被改成打开桌子 |

我的判断是：V1 的基础短句能力确实比 V0 更稳定，但还没有完全闭合。尤其是物体名词、动作和时间地点放在一起时，模型仍可能用训练中更常见的搭配替换源句内容。

### general：普通书面句

general 类整体不算理想。模型能生成看起来通顺的中文，但对英文源句的跟随不稳定。

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 19 | This article explains why clean data is important for reliable analysis. | 本 文 提 出 了 一 种 基 于 数 据 质 量 的 方 法 。 | 保留了 data quality 的方向，但原句解释关系被弱化 |
| 14 | The report describes how small changes in daily habits can improve health over time. | 这 些 新 的 研 究 表 明 ， 每 天 的 工 作 量 和 压 力 都 会 影 响 到 睡 眠 的 质 量 。 | 输出流畅，但 daily habits / health 被改写 |
| 16 | The city built a new library to give local families a quiet place to study. | 新 的 建 筑 物 ， 让 我 们 的 家 庭 成 为 一 个 新 的 城 市 。 | library 和 quiet place to study 没有保留 |

这一类说明，V1 还没有稳定掌握普通书面句的泛化翻译。它比 V0 少了一些完全无关的正式套话，但仍然容易把源句改写成更熟悉的中文表达。

### general_logic：普通逻辑句

general_logic 类能看到一个比较典型的现象：模型更容易保留逻辑连接词，但不一定能保住连接词两边的准确语义。

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 22 | If you need more information, I can send the report tomorrow. | 如 果 您 有 任 何 疑 问 ， 我 会 在 下 周 给 您 发 送 消 息 。 | if 结构保留，但 tomorrow 变成下周，report 变成消息 |
| 23 | When the meeting is over, she will call her parents. | 当 会 议 结 束 后 ， 她 会 和 她 哥 哥 说 话 。 | when 结构保留，但 parents 被替换成哥哥 |
| 29 | Although he was tired, he closed the window before leaving the room. | 虽 然 他 很 累 ， 他 还 是 会 关 上 门 。 | although 结构保留，但 window 变成 door，离开房间信息丢失 |

错误更明显的样例也不少：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 21 | Because the weather was bad, the workers moved the equipment into the room. | 因 为 天 气 很 好 ， 所 以 工 作 人 员 在 房 间 里 打 电 话 。 | bad 被反向翻成很好，动作和宾语都错 |
| 24 | Although the problem looked simple, the answer was not clear. | 虽 然 答 案 很 简 单 ， 但 问 题 很 简 单 。 | problem / answer 关系错位 |

所以这里不能只看“因为、如果、虽然”有没有出现。V1 的逻辑形式比 V0 更明显，但逻辑语义仍然不够稳，尤其是连接词两边的实体和动作容易漂移。

### technical_terms：技术术语句

technical_terms 是 V1 相比 V0 提升最明显的一类。V0 几乎无法处理 ML / Transformer 相关术语，V1 已经能在多条样例里保留核心词。

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 32 | The tokenizer converts raw text into token ids before training starts. | 在 模 型 读 取 文 本 之 前 ， 分 词 器 会 将 原 始 文 本 转 换 为 token id 。 | 基本正确，tokenizer、raw text、token id 都保留 |
| 33 | A checkpoint stores the model state so training can be restored later. | checkpoint 保 存 模 型 权 重 ， 使 训 练 之 后 可 以 继 续 。 | 基本正确，checkpoint 和恢复训练语义保留 |
| 31 | The Transformer encoder uses self-attention to read source tokens. | Transformer 编 码 器 使 用 注 意 力 掩 码 来 清 楚 表 示 词 序 。 | Transformer 和编码器保留，但 self-attention 被翻得不准 |

技术术语虽然有明显提升，但整句翻译还不稳定：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 35 | The decoder generates target tokens based on the encoder output. | 编 码 器 输 出 会 为 解 码 器 提 供 源 端 上 下 文 ， 这 样 团 队 就 能 稳 定 编 码 器 - 解 码 器 训 练 。 | 前半部分接近，但后半句添加了源文没有的训练内容 |
| 37 | The embedding layer maps each token to a dense vector. | 在 图 像 分 类 中 ， 每 个 嵌 入 层 的 嵌 入 层 。 | embedding 相关词存在，但句子不完整 |
| 40 | The optimizer updates model parameters after each batch. | 优 化 器 会 在 最 终 收 束 阶 段 使 用 优 化 器 。 | optimizer 保留，但 update parameters / batch 丢失 |

我的判断是：Stage3 到 Stage5 的技术术语数据是有效的。现在的问题不是完全不认识术语，而是术语和句内关系还没有完全绑定起来，遇到多个技术动作时容易变成模板化表达。

### tech_logic：技术逻辑句

tech_logic 比 technical_terms 更难，因为它同时要求术语、逻辑关系和源句动作都正确。V1 在这一类里有几条比较好的输出。

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 41 | If the tokenizer produces unstable token sequences, translation quality may drop. | 如 果 分 词 器 不 稳 定 ， 翻 译 质 量 可 能 会 下 降 。 | 基本正确，if 结构和核心语义保留 |
| 43 | When the learning rate is too high, validation loss may become unstable. | 当 学 习 率 过 高 时 ， 验 证 loss 可 能 会 变 得 不 稳 定 。 | 基本正确 |
| 47 | Because the encoder output contains source-side context, the decoder can follow the input sentence. | 因 为 编 码 器 先 读 取 源 端 内 容 ， 它 的 表 示 会 影 响 输 入 和 解 码 器 状 态 。 | 大意接近，encoder / decoder / source-side context 有保留 |

也有明显失败的样例：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 46 | If attention fails to connect distant tokens, the translation may lose important meaning. | 如 果 推 理 使 用 了 错 误 的 checkpoint ， 译 文 可 能 会 不 稳 定 。 | attention / distant tokens 被替换成 checkpoint 场景 |
| 48 | The tokenizer should preserve technical terms, but it also needs to handle ordinary words. | 自 动 化 应 该 是 简 单 的 ， 但 是 应 该 使 用 口 语 来 表 示 文 本 。 | tokenizer、technical terms、ordinary words 基本丢失 |

这说明 V1 已经具备一部分“技术术语 + 简单逻辑”的组合能力，但当句子里同时出现多个概念和多个动作时，模型仍会向高频技术模板靠拢。

### complex_logic：复杂逻辑句

complex_logic 仍然是当前最薄弱的部分。模型不是不能生成中文，而是不能稳定地跟住多个从句之间的关系。

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 51 | Although the validation loss decreased, the model still failed to translate part of the long sentence. | 虽 然 句 子 较 长 ， 但 验 证 loss 可 能 会 变 得 不 稳 定 ， 在 翻 译 流 程 中 。 | although 形式保留，但 decreased / failed to translate part 丢失 |
| 52 | If a sentence contains several clauses, the decoder must keep both the main idea and the condition. | 如 果 句 子 有 两 个 分 句 ， 则 应 该 保 留 一 个 句 子 ， 并 且 解 释 句 子 的 含 义 。 | 句子主干部分保留，但 main idea / condition 没有准确表达 |
| 57 | When users ask complex questions, the system should preserve the original meaning instead of generating generic answers. | 解 决 方 案 是 一 个 简 单 的 问 题 ， 因 为 它 可 以 使 用 一 个 简 单 的 方 法 来 解 决 问 题 。 | source following 明显失败，输出变成泛泛中文 |
| 60 | If the decoder follows the source sentence poorly, the output may sound natural but contain wrong information. | 如 果 解 码 器 重 复 生 成 token ， 翻 译 可 能 会 出 错 ， 但 源 端 表 示 不 会 发 生 变 化 。 | decoder / source 相关词存在，但语义被替换 |

这一类结果和训练报告里的判断一致：V1 已经把 complex_logic 纳入训练目标，但最终模型还没有把复杂逻辑句真正收住。它能保留部分关键词和逻辑形式，但长句里的限定条件、转折关系和主从句边界仍然容易丢。

### regression：历史易错点

regression 类的结果比较有参考价值，因为它能看出 replay 和 mixed regression 对基础能力有没有帮助。

表现较好的样例：

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 65 | the girl read the book in the classroom yesterday. | 女 孩 昨 天 在 教 室 里 读 这 本 书 。 | 基本正确 |
| 66 | the boy put the notebook on the desk this morning. | 男 孩 今 天 早 上 把 这 本 笔 记 本 放 在 书 桌 上 。 | 基本正确 |
| 67 | the teacher will open the door after class. | 老 师 放 学 后 会 打 开 这 扇 门 。 | 基本正确 |
| 68 | the student will close the window before class. | 学 生 上 课 前 会 关 上 这 扇 窗 户 。 | 基本正确 |

仍然有问题的样例：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 61 | open the door today. | 这 扇 门 是 在 门 边 上 的 门 边 上 。 | 祈使句退化，出现重复短语 |
| 69 | the woman moved the book next to the laptop. | 那 位 女 士 放 学 后 在 笔 记 本 电 脑 里 使 用 这 本 笔 记 本 电 脑 。 | book / laptop 混淆，动作偏移 |
| 70 | the man moved the laptop next to the book. | 那 位 男 士 放 学 后 把 这 本 笔 记 本 放 在 桌 子 上 。 | laptop / book / desk 关系错位 |

我的判断是：regression 不是没有效果，尤其是 book、notebook、door、window 放在比较常规的句子里时，模型能做对不少。但当 book / notebook / laptop 同时出现，或者句子过短、缺少上下文时，模型仍会退回到高频搭配。

### anti_pollution：抗污染样例

anti_pollution 类主要看三件事：是否出现无限重复，是否输出“正确译法是”这类规则前缀，是否把输入改写成训练报告式模板。

这次没有看到严重无限重复，也没有直接生成“正确译法是”这类固定前缀。比如：

| 编号 | EN | ZH | 观察 |
|---:|---|---|---|
| 73 | The tokenizer tokenizer tokenizer sequence should not make the output repeat the same word forever. | 分 词 器 不 是 字 典 ， 而 是 为 了 把 句 子 切 分 成 token 。 | 没有无限重复，但输出偏模板化 |
| 72 | The word Transformer should not become an ordinary electrical transformer in this machine learning sentence. | 在 Transformer 教 程 中 ， 这 个 手 写 Transformer 不 应 把 Transformer 误 译 成 无 关 的 日 常 词 。 | Transformer 保留，但输出像规则说明句 |

仍然需要注意的是，anti-pollution 还没有完全解决。下面几条虽然没有无限重复，但已经出现了比较明显的说明句倾向：

| 编号 | EN | ZH | 问题 |
|---:|---|---|---|
| 75 | Please translate the sentence naturally instead of explaining how to translate it. | 请 注 意 ， 这 个 句 子 必 须 是 自 动 写 的 。 | 出现“请注意”式说明语气 |
| 77 | The model should not output phrases like the correct translation is before giving the translation. | 翻 译 模 型 不 应 把 输 出 和 token 混 淆 ， 因 为 模 型 读 取 token 是 翻 译 模 型 的 一 部 分 。 | 没有输出指定前缀，但变成解释式技术句 |
| 78 | This input checks whether repeated technical words cause repeated Chinese output. | 这 个 手 写 Transformer 解 码 器 在 验 证 样 本 中 使 用 带 标 签 平 滑 。 | 输入检查句被改写成训练报告式表达 |

所以我会把 anti-pollution 的当前状态概括为：严重重复和固定规则前缀有所控制，但技术报告式模板输出仍然存在。

## V1 相比 V0 的推理变化

结合 V0 推理报告和这次 V1 推理结果，我认为 V1 的主要提升落在三个方面。

第一，技术术语不再整体崩溃。V0 对 `Transformer`、`tokenizer`、`checkpoint`、`learning rate`、`encoder`、`decoder` 基本没有稳定翻译能力；V1 已经能在多条样例中保留或翻译这些术语。

第二，基础短句覆盖更好。V0 能处理极短句，但稍微加入动作、时间、地点和物体组合后就容易缺失。V1 对 `the girl opened the window yesterday.`、`my brother put the notebook on the desk this morning.`、`the student will close the window before class.` 这类句子有更好的完整性。

第三，逻辑形式更容易出现。V1 在 `if`、`because`、`when`、`although` 这类句子中更容易生成“如果、因为、当、虽然”等中文结构。但这里仍然要区分“形式保留”和“语义正确”，当前 V1 还没有稳定解决后者。

## 当前主要问题

- source following 仍然不稳定。general 和 complex_logic 类里，多条输出能读，但内容已经偏离源句。
- 基础物体和动作仍会错配。book / desk、coffee / tea、chair / table、book / laptop 这类混淆在本次推理里仍然出现。
- 逻辑连接词保留不代表逻辑翻译正确。编号 21 保留了“因为”，但 bad weather 被反向翻译成“天气很好”。
- 技术句仍会模板化。模型学到了技术语境，但有时会把技术词触发成训练报告式表达。
- complex_logic 仍然没有收稳。长句、多从句、让步和条件组合时，模型容易漏掉从句关系或替换语义。
- anti-pollution 仍有残留。没有严重重复，但“请注意”“训练样本”“标签平滑”这类说明式输出还会出现。
- 当前原始输出保留 tokenizer 粒度空格，例如“女 孩 昨 天 打 开 了 这 扇 窗 户 。”。后续如果做展示，可以同时保留 raw output 和后处理版本，但需要明确区分。

## 后续计划

后续我会把 V1 推理侧继续补完整：一是把最终推理脚本、推理日志摘要、推理报告和典型样例整理到 GitHub 的 V1 目录下；二是继续围绕 source following、复杂逻辑、技术句模板化和基础物体错配做错误分析。推理策略上，后面可以继续观察 beam search、长度控制和重复惩罚对复杂句的影响。

这里不会把这次结果写成“最终完成”。更准确地说，V1 已经把基础句和技术术语能力明显推上去了，但普通书面句、复杂逻辑句和稳定跟随源句仍然是下一步要解决的问题。

## 结论

这次推理确认了 V1 curriculum 的主要训练收益：基础短句更稳定，技术术语不再像 V0 那样大面积失效，部分 tech_logic 样例可以同时保留术语和简单逻辑结构。

同时，这次推理也把当前问题暴露得比较清楚：V1 仍然会在普通书面句、复杂逻辑句和 anti-pollution 场景中出现 source following 不稳、模板化输出和局部语义替换。当前 checkpoint 可以作为 V1 阶段性成果记录，但还不能认为复杂句和通用翻译能力已经完全稳定。
