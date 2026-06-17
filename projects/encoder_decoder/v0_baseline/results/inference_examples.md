# Encoder-Decoder Transformer v0_baseline 推理示例（Epoch 20）

所有样例来自真实推理结果，按人工复核后的质量类型归类。fixed_* 为固定测试句，train / validation / test 样例来自 OPUS-100 对应 split 的随机抽样。本文件用于展示 epoch 20 baseline checkpoint 的典型能力和错误模式，不代表最终模型效果。

### 关于 fixed_* 固定测试集

fixed_simple / fixed_terms / fixed_logic 是为本轮 epoch 20 推理额外构造的固定测试集，**不来自 OPUS-100 原始 split**，目的是观察模型在训练分布以外的简单句、技术术语句和逻辑关系句上的泛化能力。

---

## 较好的案例

### 示例 G1（validation）

- **原文**: "Right here."
- **参考答案**: "就这里，快点"
- **模型预测**: "在 这 里"
- **观察**: 核心语义正确，中文自然简洁，仅语气词"快点"未翻译

---

### 示例 G2（train）

- **原文**: "Hey, what are you doing?"
- **参考答案**: "嘿，你在干什么呢？"
- **模型预测**: "嘿 ， 你 在 干 什 么 ？"
- **观察**: 完整保留了问候语、主语和动作，语义忠实

---

### 示例 G3（train）

- **原文**: "Indeed."
- **参考答案**: "的确"
- **模型预测**: "是 的"
- **观察**: 单个词的语义对应正确

---

### 示例 G4（fixed_simple）

- **原文**: "I am a student."
- **参考答案**: "我是一名学生。"
- **模型预测**: "我 是 学 生"
- **观察**: 主语、系动词、表语全部正确，缺少量词"一名"

---

### 示例 G5（fixed_simple）

- **原文**: "She likes music."
- **参考答案**: "她喜欢音乐。"
- **模型预测**: "她 喜 欢 音 乐"
- **观察**: 完整正确

---

## 部分正确的案例

> 在本次 baseline 推理中，"部分正确"的例子较为稀少。模型更常见的模式是：要么短句基本正确，要么技术句/复杂句直接失败，而非"中间态"的部分正确输出。以下选择了 4 条相对最能体现"部分正确"特征的样例。

OPUS-100 随机样本中的错误多表现为整体语义改写、实体替换或长句 hallucination，不符合 partial 的分类标准，因此本节主要使用 fixed_simple 样例，展示模型在短句上的边界能力。

### 示例 P1（fixed_simple）

- **原文**: "I love machine learning."
- **参考答案**: "我喜欢机器学习。"
- **模型预测**: "我 喜 欢 学 习 的 学 生"
- **观察**: 主语"我"和动词"喜欢"正确；但"machine learning"被拆分翻译为"学习的学生"，核心宾语丢失
- **为什么不是 good**: 宾语的语义完全错误
- **为什么仍算 partial 而非 failed**: 句子前段"我喜欢"正确保留了主谓结构，而非全句 hallucination

---

### 示例 P2（fixed_simple）

- **原文**: "We are learning English."
- **参考答案**: "我们在学习英语。"
- **模型预测**: "我 们 的 英 语"
- **观察**: "我们"和"英语"两个关键词保留；但动词"学习"丢失，从动宾结构退化为名词短语
- **为什么不是 good**: 缺少动词，句法结构不完整
- **为什么仍算 partial 而非 failed**: 两个核心名词正确，语义方向没有被重写为无关内容

---

### 示例 P3（fixed_simple）

- **原文**: "Please open the door."
- **参考答案**: "请把门打开。"
- **模型预测**: "请 打 开"
- **观察**: 动词"请打开"正确；宾语"门"完全丢失
- **为什么不是 good**: 缺宾语导致关键信息丢失
- **为什么仍算 partial 而非 failed**: 核心动作"打开"被保留，只是缺少了动作对象

---

### 示例 P4（fixed_simple）

- **原文**: "He is a teacher."
- **参考答案**: "他是一位老师。"
- **模型预测**: "他 是 一 个 教 授 。"
- **观察**: 主谓宾结构正确，但"teacher"被译成了"教授"
- **为什么不是 good**: 核心名词的语义精确度不够
- **为什么仍算 partial 而非 failed**: 整句结构和语义方向正确，仅名词存在近义偏差

---

## 较差 / 失败的案例

### 示例 F1 - 技术术语幻觉（fixed_terms）

- **原文**: "Transformer models use attention mechanisms."
- **参考答案**: "Transformer 模型使用注意力机制。"
- **模型预测**: "技 术 准 则 的 实 施 情 况 。"
- **观察**: 技术描述完全转换为 UN 风格套话，核心术语全部丢失
- **错误类型**: technical_term_error, hallucination

---

### 示例 F2 - 关键词极端偏移（fixed_terms）

- **原文**: "The training loss is decreasing."
- **参考答案**: "训练 loss 正在下降。"
- **模型预测**: "这 是 一 个 新 的 失 业 控 制 系 统 。"
- **观察**: "loss"错误映射为"失业"（unemployment），整句语义完全偏离
- **错误类型**: wrong_keyword_translation, hallucination

---

### 示例 F3 - 幻觉生成数字（fixed_terms）

- **原文**: "The model is trained on OPUS-100."
- **参考答案**: "模型在 OPUS-100 上训练。"
- **模型预测**: "这 个 网 络 的 成 本 估 计 为 1, 000 万 美 元 。"
- **观察**: 凭空生成虚构数字，与原文毫无关系
- **错误类型**: hallucination

---

### 示例 F4 - 逻辑句语义偏移（fixed_logic）

- **原文**: "The source sentence is simple, so the prediction should be stable."
- **参考答案**: "源句很简单，所以预测应该稳定。"
- **模型预测**: "这 是 一 个 可 预 测 的 解 决 办 法 ， 不 可 避 免 的 。"
- **观察**: 单句逻辑关系词"so"被丢失，"因果"变为了"判断"
- **错误类型**: logic_relation_loss, wrong_meaning

---

### 示例 F5 - 逻辑句术语错误（fixed_logic）

- **原文**: "When the learning rate is too high, training may become unstable."
- **参考答案**: "当学习率过高时，训练可能变得不稳定。"
- **模型预测**: "在 这 方 面 ， 学 生 的 比 率 很 高 。"
- **观察**: "learning rate"被译成"学生比率"，时间连接词"when"变为"在这方面"
- **错误类型**: technical_term_error, logic_relation_loss

---

### 示例 F6 - 数字和时间全部错误（test）

- **原文**: "Eight new individuals and six new entities have been placed on the Committee's List since 25 May 2004."
- **参考答案**: "自2004年5月25日以来，又有8名个人和6个实体列入委员会清单。"
- **模型预测**: "2003 年 1 月 ， 委 员 会 在 其 网 站 上 发 表 了 一 份 声 明 ， 其 中 包 括 ："
- **观察**: 仅"委员会"一词保留，所有数字和时间均被替换
- **错误类型**: wrong_number, missing_information

---

### 示例 F7 - 实体替换（train）

- **原文**: "Belgrade and Pristina will focus on developing the special nature of the relations existing between them especially in their historical, economic, cultural and human dimension."
- **参考答案**: "1. 贝尔格莱德和普里什蒂纳将聚焦于发展双方关系的特殊性质，特别是这一关系的历史、经济、文化和人文层面。"
- **模型预测**: "这 些 都 是 在 科 索 沃 和 梅 托 希 亚 的 经 验 中 ， 它 们 的 目 标 是 ：..."
- **观察**: 原文地名"贝尔格莱德和普里什蒂纳"被替换为"科索沃和梅托希亚"
- **错误类型**: wrong_entity, hallucination

---

### 示例 F8 - 口语表达完全错译（train）

- **原文**: "Whose money, man?"
- **参考答案**: "谁的钱，伙计？"
- **模型预测**: "谁 是 老 大 ？"
- **观察**: 仅疑问词"谁"保留，其余语义全部改变
- **错误类型**: wrong_meaning

---

### 示例 F9 - 人名幻觉（train）

- **原文**: "Jose Martinez?"
- **参考答案**: "Jose Martinez?"
- **模型预测**: "玛 丽 ・ 克 莱 克 斯 ？"
- **观察**: 常见西语人名被完全替换为无关中文名
- **错误类型**: wrong_entity, hallucination

---

> **关于部分正确样例偏少的说明**：本轮 v0_baseline 推理中，模型呈现明显的两端分化——简单句多可直接翻译，技术句和复杂句多直接失败。真正的"部分正确中间态"确实较少，这反映出当前 baseline 尚未建立领域外词汇的渐进式语义映射，后续训练优化将重点关注这一问题。
