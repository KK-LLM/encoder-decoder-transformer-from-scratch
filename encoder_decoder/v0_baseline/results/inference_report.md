# Encoder-Decoder Transformer v0_baseline 推理报告（Epoch 20）

## 概述

- **项目版本**：Encoder-Decoder Transformer v0_baseline
- **Checkpoint**：`checkpoint_epoch_20.pt`（epoch=20）
- **Tokenizer**：训练后保存的 tokenizer（vocab_size=16000）
- **任务**：English → Chinese 翻译
- **报告目的**：评估 epoch 20 checkpoint 的阶段性推理效果，为下一轮训练策略提供依据

---

## 评估设置

| 项目 | 值 |
|---|---|
| checkpoint 路径 | `train_checkpoint_opus100_en_zh/checkpoint_epoch_20.pt` |
| tokenizer 路径 | `train_checkpoint_opus100_en_zh/tokenizer` |
| 训练集来源 | `opus100_en_zh_local/train`（数据目录约 1,000,000 条；本次 v0_baseline 训练实际使用前 500,000 条） |
| 验证集来源 | `opus100_en_zh_local/validation`（2,000 条） |
| 测试集来源 | `opus100_en_zh_local/test`（2,000 条） |
| train 抽样 | 50 条（seed=42 随机抽样） |
| validation 抽样 | 100 条（seed=42 随机抽样） |
| test 抽样 | 100 条（seed=42 随机抽样） |
| 固定测试集 | fixed_simple（15，简单句）、fixed_terms（20，技术术语）、fixed_logic（20，逻辑关系），均为额外构造，不来自 OPUS-100 split |
| decode 策略 | greedy decoding |
| max_src_len / max_tgt_len | 96 / 96 |
| 运行设备 | Apple Silicon MPS（CPU fallback on non-CUDA） |
| BLEU / chrF | 未计算（环境未安装 sacrebleu） |

**遇到的问题**：推理脚本中模型类属性命名（`self_attn_conn` vs `self_attn_connection`）与 checkpoint 中的 key 不匹配，已在新创建的 `inference_eval_script.py` 中修正为与训练脚本完全一致的命名。

---

## 训练日志摘要

基于训练日志 `train_logs/logs/opus_train_log.log`：

- **epoch 范围**：1–48
- **train_loss 趋势**：5.9811（epoch 1） → 2.9984（epoch 48），持续下降
- **valid_loss 趋势**：4.5595（epoch 1） → 3.4693（epoch 20, **最低**） → 3.5653（epoch 48）
- **epoch 20 位置**：valid_loss 全局最低点（3.4693）
- **过拟合判断**：epoch 20 后 train_loss 继续下降（3.2443 → 2.9984），但 valid_loss 波动上升，train/valid gap 从 0.23 扩大至 0.57，呈现典型过拟合趋势
- **选用 epoch 20 的原因**：valid_loss 最低，泛化指标最优

---

## 固定句测评结果

### 简单句稳定性

| 英文输入 | 模型输出 | 分析 |
|---|---|---|
| This is a book. | 这 是 书 | 基本正确，缺少量词"一本" |
| I am a student. | 我 是 学 生 | 正确 |
| He is a teacher. | 他 是 一 个 教 授 。 | 基本正确，teacher 译成"教授"不够准确 |
| She likes music. | 她 喜 欢 音 乐 | 正确 |
| I want to drink water. | 我 要 喝 杯 水 | 正确 |
| The weather is good today. | 今 天 真 是 太 好 了 | 意思接近，但语法松散 |
| I have a question. | 我 有 问 题 | 正确 |
| Please open the door. | 请 打 开 | 缺少宾语"门" |
| I love machine learning. | 我 喜 欢 学 习 的 学 生 | machine learning 被拆分翻译，语义扭曲 |
| This model is small. | 这 是 一 个 很 好 的 。 | 语义完全偏离 |
| The training is finished. | 这 里 有 大 学 | 完全错译 |
| The result is correct. | 这 是 一 个 重 要 的 决 定 。 | 语义偏离 |
| The sentence is short. | 现 在 不 是 时 候 了 | 完全错译 |
| The model can translate simple sentences. | 我 们 的 选 择 是 不 同 的 | 完全错译，幻觉 |
| We are learning English. | 我 们 的 英 语 | 近似正确但缺少动词 |

**小结**：
- 极简主谓宾句（"I am a student."、"She likes music."）翻译基本正确
- 包含"model"、"training"、"result"等 OPUS100 中未以"技术含义"出现的词时，极易出现语义偏移
- 15 条中有 6 条基本可接受（约 40%），剩余出现语义丢失或显著错误
- 原因：训练集中几乎没有"日常 + 技术"混合句，模型学到的"model"映射为 UN 语境下的"模式/模型"，"training"映射为"培训/训练"，而非 ML 语境

### 专业名词翻译

| 英文输入 | 模型输出 | 分析 |
|---|---|---|
| I love machine learning. | 我 喜 欢 学 习 的 学 生 | machine learning → "学习的学生" |
| Machine learning is useful. | 学 习 的 机 会 | 完全错译 |
| Deep learning is a part of machine learning. | 学 生 的 作 用 很 好 | 完全错译 |
| Transformer models use attention mechanisms. | 技 术 准 则 的 实 施 情 况 。 | 完全错译，"Transformer"无法识别 |
| The encoder reads the source sentence. | 在 这 个 时 候 ， 我 们 的 名 字 是 ： | 完全幻觉 |
| The decoder generates the target sentence. | 这 是 一 个 战 争 的 地 方 | 完全幻觉 |
| The checkpoint is saved after training. | 这 是 一 个 训 练 有 素 的 工 作 人 员 。 | "checkpoint"误译，"training"译为"训练有素" |
| We load the checkpoint for inference. | 我 们 的 工 作 计 划 是 在 这 里 。 | 完全错译 |
| The tokenizer converts text into tokens. | 这 是 一 个 新 的 方 法 ， 可 以 在 下 面 。 | 完全错译 |
| The model uses positional encoding. | 这 是 一 个 可 以 接 受 的 技 术 。 | 完全错译 |
| The attention layer computes token relationships. | 这 是 一 个 有 意 义 的 调 查 。 | 完全错译 |
| The training loss is decreasing. | 这 是 一 个 新 的 失 业 控 制 系 统 。 | loss → "失业"，极度偏离 |
| The validation loss is important. | 这 是 一 个 严 重 的 损 失 。 | loss → "损失"，偏离 |
| The learning rate schedule improves training stability. | 培 训 课 程 改 革 。 | learning rate → "培训课程" |
| The batch size affects GPU memory usage. | " 我 们 的 飞 机 " | 完全幻觉 |
| The source sentence is in English. | 英 文 本 中 的 一 部 分 书 名 | 局部正确（"英文"）但整体偏离 |
| The target sentence is in Chinese. | 最 后 一 个 国 家 | 完全错译 |
| The model is trained on OPUS-100. | 这 个 网 络 的 成 本 估 计 为 1, 000 万 美 元 。 | 幻觉生成数字 |
| Beam search may improve translation quality. | 需 要 更 多 的 电 视 。 | 完全错译 |
| Greedy decoding selects the most likely token. | 有 人 说 ， 这 是 一 个 很 大 的 房 子 。 | 完全幻觉 |

**小结**：
- 20 条技术句全部失败，无一条能正确保留或翻译核心术语
- "machine learning"、"Transformer"、"encoder"、"decoder"、"tokenizer"、"checkpoint"、"attention"、"loss"等术语全部无法正确翻译
- 多句出现严重幻觉（生成与原文无关的虚构内容）
- 根因：本次训练数据以 OPUS100 正式文本为主，未覆盖 ML/CS 领域语境。进一步检查 tokenizer 后发现，术语切分并不统一：例如 "loss" 和 "attention" 可作为单一 token 出现，而 "Transformer"（3 subword）、"checkpoint"（3 subword）、"tokenizer"（4 subword）则被拆分为多个 subword token（fixed_* 测试涉及的全部 ML/CS 术语切分结果见 `infer_logs/logs/tokenizer_term_check_epoch20.txt`）。结合推理结果看，术语失败不仅是 tokenizer 切分问题，更主要是因为训练语料缺少对应领域语境。

### 逻辑关系句翻译

| 英文输入 | 模型输出 | 分析 |
|---|---|---|
| If the model trains longer, the loss may become lower. | 如 果 能 够 控 制 下 的 速 度 ， 就 可 能 会 有 一 个 可 能 性 。 | "如果...就"结构保留，但语义完全偏离 |
| If the validation loss increases, the model may be overfitting. | 如 果 是 这 样 的 话 ， 就 会 造 成 损 失 。 | "如果...就"结构保留，语义偏离 |
| Although the sentence is short, the translation is wrong. | 但 是 ， 这 是 一 个 不 公 平 的 审 判 。 | "虽然"变"但是"，语义完全偏离 |
| Although the model can translate simple sentences, it still fails on long sentences. | 但 是 ， 这 是 一 个 不 同 的 时 间 ， 不 过 ， 它 的 规 定 是 一 个 更 好 的 方 法 。 | "虽然"丢失，但"但是/不过"出现，语义混乱 |
| Because the dataset is noisy, the model may learn wrong patterns. | 这 是 一 个 不 同 的 数 据 ， 因 为 它 是 一 个 概 念 ， 不 是 一 个 概 念 。 | "因为"被保留，但语义混乱，出现逻辑重复 |
| Because the tokenizer is important, we save it after training. | 这 是 我 们 的 工 作 ， 因 为 我 们 需 要 一 个 新 的 警 察 。 | "因为"保留，但语义幻觉（"警察"） |
| The source sentence is simple, so the prediction should be stable. | 这 是 一 个 可 预 测 的 解 决 办 法 ， 不 可 避 免 的 。 | "因此"等价词未出现，因果丢失 |
| The model remembers some training examples, but it does not generalize well. | 不 过 ， 这 些 数 字 是 一 个 很 好 的 例 子 ， 不 过 ， 有 一 个 数 字 ， 有 一 个 更 好 的 主 题 。 | "但是" → "不过"，保留但语义完全偏离 |
| The output is fluent, but the meaning is not correct. | 不 过 ， 这 是 一 个 很 好 的 问 题 ， 不 过 ， 这 个 问 题 是 不 同 的 。 | "但是" → "不过"，保留但语义混乱 |
| If beam search is added, the translation may become more stable. | 如 果 有 人 选 择 ， 那 么 就 可 以 更 好 地 选 择 。 | "如果"保留，语义偏离 |
| If the checkpoint is not loaded correctly, inference will fail. | 如 果 没 有 ， 就 没 有 任 何 证 据 了 | "如果"保留，语义完全不同 |
| When the learning rate is too high, training may become unstable. | 在 这 方 面 ， 学 生 的 比 率 很 高 。 | "当"丢失，learning rate → "学生比率" |
| The model performs better on the training set than on the test set. | 电 脑 化 的 工 作 量 是 一 个 不 同 的 数 据 。 | 比较结构完全丢失 |

**小结**：
- 部分逻辑连接词（"如果...就"、"因为"、"但是/不过"）在语法层面被保留，但语义内容几乎全部错误
- 技术内容无法翻译的前提下，逻辑结构即使被保留也变得无意义——形式对但内容错
- 比较关系（"better than"）全部丢失
- "when"、"so" 等逻辑词在多数情况下未被正确翻译
- 如果源句的核心名词能被翻译（如日常句），逻辑结构更可能被保留；否则逻辑符号沦为空洞语法框架

---

## 定性示例

### 相对较好的案例

#### Case 1 - 简短对话（validation）

- **原文**: "Right here."
- **参考答案**: "就这里，快点"
- **模型预测**: "在 这 里"
- **观察**: 核心语义正确，中文自然简洁
- **错误类型**: correct_or_near_correct
- **分析**: 极短句和常见表达能正确翻译，说明模型对高频短句已建立有效映射

### 语体流畅但语义不忠实的案例

以下案例输出流畅、句法正确且符合训练数据的文体风格，但语义与原文存在明显偏差。单独列出是为了说明当前 baseline 的一个典型特征：模型可以学会目标语体的生成模式，但尚未建立稳定、精确的语义映射。

#### Case 2 - 联合国文档句式（train）

- **原文**: "National measures had been taken to combat desertification in the last four decades and further work was urgently required."
- **参考答案**: "过去四十年里为防治荒漠化采取了国家措施，但迫切需要开展进一步的工作。"
- **模型预测**: "在 这 方 面 ， 我 们 强 调 ， 在 2002 年 以 前 ， 在 森 林 覆 盖 面 上 的 最 低 限 度 的 干 旱 和 消 除 贫 穷 仍 然 是 一 个 严 重 的 挑 战 。"
- **观察**: 输出是联合国风格的流畅中文，句式正确。但内容与原文不完全匹配，更像是意译/自由生成。
- **错误类型**: hallucination（语义偏离但句法正确）
- **分析**: 模型学会了联合国文体风格，但语义精确度不足。这说明模型更像在"生成类似风格的句子"而非"精确翻译"

#### Case 3 - 正式通知（test）

- **原文**: "Delegations are requested to submit the names of their representatives to the Secretary of the Preparatory Committee, Ms..."
- **参考答案**: "请各代表团将其代表姓名送交给筹备委员会秘书Vivian Pliner-Josephs女士（S-2950E室；电传：(212)963-5935）。"
- **模型预测**: "请 各 代 表 团 注 意 ， 应 将 代 表 团 的 姓 名 提 交 给 第 六 委 员 会 秘 书 Jennifer de Laurentis 女 士 （ 电 话 ： 1 （ 212 ） 963 - 5935 ； 传 真 ： 1 （ 2"
- **观察**: 句式正确传达了请求的含义，联系方式等细节部分保留；但人名和委员会编号出现替换
- **错误类型**: untranslated_english, hallucination
- **分析**: UN 文档格式句处理较好，但专有名词的精确重现能力不足

### 部分正确的案例

#### Case 4 - 口语短句（test）

- **原文**: "Hey, sir! Two more please!"
- **参考答案**: "嗨，服務員，再來兩杯"
- **模型预测**: "嘿 ， 先 生 们 ， 等 等 ！"
- **观察**: "Hey" → "嘿" 正确，"sir" → "先生们" 半正确，"Two more please" → "等等" 严重错误
- **错误类型**: missing_information, wrong_meaning
- **分析**: 口语短句的部分成分翻译正确，但整体语义重组失败。"再要两杯"变成了"等等"，说明模型对非正式表达的语义理解有限

#### Case 5 - 法律/联合国文本（validation）

- **原文**: "Do you think that we would ever go to jail if we were caught?"
- **参考答案**: "你认为 我们会永远去坐牢 如果我们抓住了？"
- **模型预测**: "你 们 要 不 要 我 们 去 看 看 那 个 死 者 ？"
- **观察**: 疑问句结构保留，但核心语义从"坐牢"变成了"看死者"——greedy decoding 在语义上"脑补"了完全不同的事件
- **错误类型**: hallucination, wrong_meaning
- **分析**: 模型倾向用一个"可能的句式"而非"准确的语义"来生成，展现出基于统计共现的生成行为

#### Case 6 - 正式表述（train）

- **原文**: "Belgrade and Pristina will focus on developing the special nature of the relations existing between them..."
- **参考答案**: "1. 贝尔格莱德和普里什蒂纳将聚焦于发展双方关系的特殊性质，特别是这一关系的历史、经济、文化和人文层面。"
- **模型预测**: "这 些 都 是 在 科 索 沃 和 梅 托 希 亚 的 经 验 中..."
- **观察**: 地域相关词汇被部分捕获（"科索沃"），但原文的地名"贝尔格莱德和普里什蒂纳"完全丢失
- **错误类型**: wrong_keyword_translation, missing_information
- **分析**: 专有名词翻译不稳定，模型倾向于用见过的高频相关词替代

#### Case 7 - 项目描述（test）

- **原文**: "The strategies include capacity-building for human resources in the states and municipalities for implementing the Acute Diarrhoeal Diseases Monitoring..."
- **参考答案**: （长参考：能力建设、监测方案、技术支持、手册修订等）
- **模型预测**: "国 家 一 级 的 方 案 和 项 目 ， 包 括 ： 监 测 和 评 价 国 家 发 展 计 划..."
- **观察**: 句首概括结构（"包括"）和部分关键词（"监测"）正确，但内容大幅改写
- **错误类型**: under_translation, hallucination
- **分析**: 长句处理时模型倾向"抓结构、放细节"，输出流畅但语义丢失

### 较差 / 失败的案例

#### Case 8 - 技术句（fixed_terms）

- **原文**: "Transformer models use attention mechanisms."
- **模型预测**: "技 术 准 则 的 实 施 情 况 。"
- **观察**: 一整句技术描述变成了完全无关的 UN 风格表述
- **错误类型**: wrong_meaning, technical_term_error
- **分析**: 模型只在训练数据领域内运作，无法泛化到技术英语。这是当前 baseline 最核心的限制。

#### Case 9 - 技术句（fixed_terms）

- **原文**: "The training loss is decreasing."
- **模型预测**: "这 是 一 个 新 的 失 业 控 制 系 统 。"
- **观察**: "training loss" → "失业"（unemployment），极端语义偏移
- **错误类型**: wrong_keyword_translation, hallucination
- **分析**: 经 tokenizer 检查，"loss" 确实被切分为单一 token，训练数据中 "loss" 几乎只出现在 "job loss"、"weight loss" 等非 ML 语境，模型没有机会学到 ML 领域下的 "loss" 语义

#### Case 10 - 重复句式（train）

- **原文**: "I also warned that further growth of the trial docket would make achieving that ambitious target entirely dependent on at least some cases being disposed..."
- **模型预测**: "我 们 认 为 ， 在 审 判 期 间 ， 这 些 案 件 的 审 判 工 作 不 仅 仅 是 在 一 个 案 件 中 ， 而 且 还 是 在 一 个 案 件 中 ， 而 且 还 是 在 一 个 案 件 中..."
- **观察**: 输出出现语义重复（"案件"多次循环）和句式冗余，虽然未触发 hard repeat 规则但明显质量差
- **错误类型**: repeated_tokens（轻度）, too_long
- **分析**: Greedy decoding 的 argmax 策略在复杂句上容易陷入局部重复模式

#### Case 11 - 人名词（train）

- **原文**: "Jose Martinez?"
- **参考答案**: "Jose Martinez?"
- **模型预测**: "玛 丽 ・ 克 莱 克 斯 ？"
- **观察**: 人名 Jose Martinez（常见西语名）被完全替换为无关人名
- **错误类型**: wrong_keyword_translation, hallucination
- **分析**: 罕见人名无法保留，模型倾向于用高频中文名替代

#### Case 12 - 口语表达（train）

- **原文**: "Whose money, man?"
- **参考答案**: "谁的钱，伙计？"
- **模型预测**: "谁 是 老 大 ？"
- **观察**: "谁"保留，但问句意思完全改变
- **错误类型**: wrong_meaning
- **分析**: 口语化表达是训练数据中的稀缺样本，模型用统计上最常见的疑问句式替代

---

## 错误分析

| 错误类型 | 描述 | 推理中的证据 | 可能原因 | 下一轮训练改进 |
|---|---|---|---|---|
| **technical_term_error（专业术语错误）** | ML/CS 术语全部无法翻译 | Transformer→"技术准则"、loss→"失业"、checkpoint 无法识别 | 训练数据以 OPUS100 正式文本为主，未覆盖 ML 语境；经实际检查，术语切分不统一：loss 为单一 token，Transformer/checkpoint/tokenizer 则为 3-4 个 subword token | 在训练集中加入 ML 术语双语样本，置于自然完整句中训练 |
| **hallucination（幻觉）** | 生成与源句语义无关的流畅中文 | "batch size"→"我们的飞机"、"OPUS-100"→"1000万美元" | 模型在语义不确定时倾向生成为训练数据中最常见的句式 | 增加更多样化的训练数据 |
| **domain_overfit（领域过拟合）** | 模型只能处理 UN 风格文本，日常和技术英语表现极差 | fixed_simple 仅 40% 可接受，fixed_terms 0%，fixed_logic 全失败 | 训练数据 100% 来自 OPUS100 正式语体 | 混合多种语体的训练数据（日常对话、技术文档、新闻等） |
| **long_sentence_degradation（长句退化）** | 长句翻译语义丢失或部分重组 | 多数 >50 词句子输出虽流畅但语义不忠实 | 模型容量有限，长距离依赖捕捉不足；greedy 解码加剧 | 增加长句训练样本 |
| **wrong_keyword_translation（关键词错译）** | 具体名词/专有名词错译 | "Belgrade"→"科索沃"、"Jose Martinez"→"玛丽·克雷克斯" | 罕见实体在训练中曝光不足，模型用高频近似词替代 | 数据增强（实体替换）；提升数据多样性 |
| **repeated_generation（重复生成）** | 复杂句上出现语义循环或重复句式 | "案件"循环、"而且...而且..." 冗余 | greedy decoding 在低概率区域倾向重复 high-prob token | 在训练数据中增加更多样化的句式结构 |
| **output_too_long（输出过长）** | 预测长度 ~2× 参考长度 | 长度比 1.9-2.0，15-20% 样本过长 | EOS 生成偏保守；训练数据以长句为主 | 检查训练数据长度分布；调整 label smoothing |
| **logic_relation_loss（逻辑关系丢失）** | 逻辑关系词保留但语义全错 | "如果...就"结构保留但内容不匹配 | 模型学会了句法模板但语义映射不准 | 增加逻辑关系句专项训练数据 |
| **data_noise（数据噪声）** | 参考翻译本身存在质量问题 | 部分参考为逐词直译、语序不自然 | OPUS100 数据集特点，不完全是翻译问题 | 无需针对此训练，但评估时应认知到 |

---

## 泛化分析

1. **train vs validation/test**：三者质量分布接近，未观察到 train 显著好于 test 的强过拟合信号。但 train 平均参考长度（29.9）明显短于 valid（53.8），说明随机抽样导致了分布差异，不应直接比较。

2. **Validation vs test**：二者在长度比和质量分布上高度接近，说明评估结果相对稳定，test 集的表现可代表模型在该语体上的泛化能力。

3. **过拟合还是欠拟合**：从 valid loss 曲线看，epoch 20 之后出现明显过拟合趋势（valid loss 上升且 train loss 继续下降）。但从推理质量看，当前模型更接近"领域过拟合"——即模型学到了 OPUS100 UN 语体的统计特征，但在语体风格上几乎是"零样本"泛化（对日常英语和技术英语完全失败）。

4. **简单句稳定性**：极简主谓宾句基本稳定（"I am a student."→"我是学生"），但包含未见领域词汇的短句容易出错（"The training is finished."→"这里有大学"）。

5. **专业名词稳定性**：完全不稳定，20/20 全部失败。

6. **逻辑关系稳定性**：语法层面部分保留（"如果"、"但是"等），但语义内容错误时逻辑结构变得无意义。

7. **当前 checkpoint 是否适合作为下一轮优化起点**：适合。epoch 20 的 valid_loss 是全局最低，推理层面也证明模型具备基础解码能力。在此基础上扩展训练数据（增加多语体样本）比从零训练更高效。

---

## 当前限制

- **领域单一**：v0_baseline 仅在 OPUS100 en-zh UN 语体上训练，技术术语翻译能力接近零，日常英语稳定性严重不足。这是训练数据分布的反映，而非模型结构缺陷。
- **简单句稳定性不足**：极简主谓宾句尚可，但包含"model"、"training"等未见领域词汇时极易触发语义偏离或幻觉。这在本报告的 fixed_simple 测试中直接暴露。
- **输出偏长**：所有数据子集上预测长度约为参考长度的两倍（长度比 1.9–2.0），EOS 生成偏保守。
- **tokenizer 词表偏小**：vocab_size 仅 16000，技术术语（如 Transformer、checkpoint）被切分为碎片化 token，难以建立稳定的语义映射。
- **缺乏系统性错误监控**：固定测试集仅 55 条，无法覆盖 easy daily、error regression、tech sentence 和 anti-pollution 等多维评估需求。
- **checkpoint 不完整**：当前 checkpoint 仅包含 model_state_dict，不含 optimizer/scheduler/RNG 状态，无法无缝恢复原训练状态；但可加载 model_state_dict 作为继续微调或阶段训练的初始化权重。

---

## 基于本轮推理的下一轮优化方向

本轮推理结果的核心启示是：对从零训练的手写 Encoder-Decoder Transformer 来说，**基础翻译数据的质量和覆盖面比单纯堆参数或复杂数据更重要**。下一轮 v1 的核心改进方向：

- **采用分阶段学习策略**：不一次性混合全量数据，而是分阶段从基础短句渐进过渡到技术句和复杂逻辑句，后期用修复型训练平衡各类能力
- **扩展模型与词表**：d_model 升级至 768、层数增至 10、tokenizer vocab 扩展至 48000，提升模型容量和术语切分精度
- **针对性构造高质量数据**：本轮暴露的 door/window 混淆、Transformer 漏译、loss→"失业"等典型错误将直接转化为 error regression pairs；同时加入 anti-pollution 样本防止训练后期污染输出
- **建立多维固定评估集**：扩展至 500 条，覆盖 easy daily、time/action/object、error regression、tech sentence 和 anti-pollution 五个维度，支撑跨版本纵向对比
- **保存完整 checkpoint**：含 optimizer/scheduler 状态，支持从任意阶段恢复训练，服务于分阶段多轮续训需求

---

## 总结

当前 Encoder-Decoder Transformer v0_baseline 的 epoch 20 checkpoint 证明：

1. **训练和推理闭环已跑通**：从训练脚本、tokenizer 训练、checkpoint 保存到推理脚本，完整链路工作正常
2. **模型具备基础翻译能力**：在 OPUS100 UN 语体内的句子可以生成流畅、部分忠实的中文，零空输出，零特殊 token 污染
3. **泛化能力严重受限于训练数据**：技术英语（ML/CS 术语）和日常英语几乎完全无法翻译，这是训练数据单一的直接结果，而非模型结构的根本问题
4. **过拟合趋势已出现**：epoch 20 后 valid loss 上升，训练数据记忆能力在提升但泛化停滞
5. **本轮推理分析为下一轮优化提供了明确方向**：采用分阶段学习策略、针对性构造 error regression 和技术术语自然句、以及建立固定评估集是下一轮最优先的改进手段
