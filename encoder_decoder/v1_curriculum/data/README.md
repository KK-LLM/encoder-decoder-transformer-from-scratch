# v1_curriculum Data

该目录保存 `v1_curriculum` 课程学习训练所需的数据文件。

## Files

| 文件 | 行数 | 用途 |
|------|------:|------|
| `stage1/train.jsonl` | 107,000 | Stage1 基础短句训练 |
| `stage2/train.jsonl` | 281,681 | Stage2 日常句、普通书面句和简单逻辑训练 |
| `stage3/train.jsonl` | 432,987 | Stage3 技术句和术语稳定训练 |
| `stage4/train.jsonl` | 503,000 | Stage4 综合逻辑和技术逻辑训练 |
| `stage5/train.jsonl` | 548,084 | Stage5 最终收束、回放、防遗忘和 anti-pollution 训练 |
| `final_eval.jsonl` | 15,000 | Stage1-Stage5 统一验证集 |

## JSONL Schema

每一行是一个英文到中文翻译样本：

```json
{"source": "Please open the window in the room before work.", "target": "请上班前在房间里打开这扇窗户。"}
```

字段说明：

| 字段 | 说明 |
|------|------|
| `source` | 英文输入句子 |
| `target` | 中文目标翻译 |

## Validation Policy

本次分阶段训练统一使用：

```text
final_eval.jsonl
```

作为 Stage1 到 Stage5 的 valid loss 验证集。

每个 stage 的本地目录中可能存在 `test.jsonl`，但这些文件不上传到 GitHub，也不用于本次分阶段训练的 valid loss 计算。这样做是为了保证不同 stage 的 valid loss 可以在同一个验证集上纵向比较。

## Stage Design

### Stage1

Stage1 是基础短句阶段，重点建立最基础的英中词序映射、常见动作、常见物体、时间地点、人物关系和简单日常句。

主要类别包括：

- easy_basic
- basic_daily_sentence
- simple_action_sentence
- simple_object_sentence
- simple_time_sentence
- simple_place_sentence
- simple_person_sentence
- small clean general_written

### Stage2

Stage2 是基础能力扩展阶段，在 Stage1 基础上扩大日常句、普通书面句、简单逻辑句和基础错误回归样本。

主要类别包括：

- easy_basic replay
- medium_daily_sentence
- general_written
- simple_logic_sentence
- error_regression_basic
- object_confusion_repair
- time_expression_repair
- basic_general_translation

### Stage3

Stage3 是技术句与术语稳定阶段，重点让模型接触 ML / DL / Transformer / tokenizer / encoder / decoder 等技术语境。

主要类别包括：

- tech_sentence
- terminology
- real_tech_sentence
- simple_logic_with_tech
- general_translation_replay
- error_regression_tech
- terminology_stability
- Stage1 / Stage2 clean replay

术语训练使用自然完整句作为载体，而不是词典条目或规则说明句。重点术语包括：

- machine learning
- deep learning
- Transformer
- tokenizer
- encoder
- decoder
- checkpoint
- learning rate
- loss
- epoch
- dataset
- training
- inference
- attention
- embedding
- optimizer
- gradient
- batch
- validation
- evaluation

### Stage4

Stage4 是综合逻辑强化阶段，重点加强普通逻辑、复杂逻辑、书面表达、技术逻辑混合句和 mixed regression。

主要类别包括：

- general_written_replay
- easy_replay
- general_logic
- complex_logic
- tech_logic_sentence
- error_regression_mixed
- simple_logic_with_tech
- mixed_regression
- anti_forgetting replay

Stage4 的目标不是单纯增加句子难度，而是把普通逻辑、复杂逻辑和技术语境结合起来。

### Stage5

Stage5 是最终收束阶段，重点是综合回放、防遗忘、逻辑句稳定、技术术语稳定、complex_logic 收束、anti-pollution 和 mixed_regression。

主要类别包括：

- easy_replay
- general_written_replay
- tech_sentence_replay
- terminology_stability
- complex_logic
- tech_logic_sentence
- mixed_regression
- anti_forgetting replay
- anti_pollution stability
- encoder / decoder / tokenizer / checkpoint / validation loss 等技术语境句
- 长句和多句组合样本

Stage5 不继续盲目增加难度，而是让基础句、技术句和复杂逻辑句之间取得最终平衡。

## Data Sources

v1 的 stage 数据不是单一数据集训练，也不是把公开语料直接拼接。构造前先建立了本地统一数据池：

```text
/Users/xulinqing/Pycharm_Project/annotated-transformer-master/available_data_unified
```

该目录中的数据来自多个公开来源，后续再经过筛选、清洗、去重和课程化重组，才进入 Stage1 到 Stage5。

### Source Inventory

| 来源类型 | 数据集 / 子目录 | 原始来源 | 在课程数据中的主要作用 |
|---|---|---|---|
| OPUS 官方语料 | `opus_ccaligned`, `opus_ccmatrix`, `opus_paracrawl` | OPUS / object.pouta.csc.fi | 大规模 web-mined 平行语料，主要作为 general written / logic 候选池，严格过滤后少量使用 |
| OPUS 官方语料 | `opus_globalvoices`, `opus_wikimedia`, `opus_wikititles` | OPUS GlobalVoices / Wikimedia / WikiTitles | 新闻、社区文章、Wikimedia 和标题类平行文本，用于普通书面句和实体表达 |
| OPUS 官方语料 | `opus_gnome`, `opus_kde4`, `opus_php`, `opus_ubuntu` | OPUS GNOME / KDE4 / PHP / Ubuntu | 软件本地化和技术文档语料，用于技术词、界面词和短句翻译候选 |
| OPUS 官方语料 | `opus_opensubtitles`, `opus_qed`, `opus_tatoeba`, `opus_tico19`, `opus100_en_zh` | OPUS OpenSubtitles / QED / Tatoeba / TICO-19 / OPUS-100 | 口语、教育、句子级平行语料和医学/COVID 语料，用于日常句、短句、普通翻译和对照测试 |
| ModelScope | `modelscope_wmt_zh_en` | ModelScope WMT Chinese-English machine translation corpus | 大规模 WMT 风格中英语料；因部分中文存在分词空格和噪声，只在清洗后谨慎使用 |
| WMT / 新闻语料 | `wmt19_zh_en`, `wmt_news` | WMT19 zh-en streaming sample / WMT news style local corpus | 新闻、评论、正式书面句和逻辑句候选，需限制比例避免语体过重 |
| Hugging Face datasets | `wikimatrix_en_zh`, `pawsx`, `mlqa`, `finetranslations_edu_optional` | Hugging Face streaming / dataset snapshots | WikiMatrix 平行句、PAWS-X/XNLI/MLQA 派生样本和教育语料候选；不适合直接全量混入，按任务类型抽取 |
| GitHub / benchmark | `hardmtbench`, `flores200`, `tatoeba_challenge` | GitHub: HardMTBench, facebookresearch/flores, Helsinki-NLP/Tatoeba-Challenge | hard domain benchmark、FLORES dev/devtest、Tatoeba Challenge，用于评估意识和高质量样本参考 |
| 官方网站下载 | `tatoeba_official`, `tico19`, `cc_cedict`, `kaikki_chinese` | Tatoeba downloads, TICO-19, MDBG CC-CEDICT, Kaikki/Wiktionary | 句子级平行语料、医学 benchmark、词典/术语资源；词典类不作为普通平行句直接堆入 |
| 技术文档 / GitHub 文档 | `tensorflow_docs_l10n`, `kubernetes_website_l10n`, `hf_blog_translation`, `huggingface_course_translation` | TensorFlow / Kubernetes / Hugging Face 文档与博客翻译资源 | 技术说明句、文档句、术语上下文和 anti-pollution 检查来源，需要强对齐检查 |
| 专业技术资料 | `professionally_d2l_en_zh`, `professionally_huggingface_course`, `professionally_pytorch_docs_zh`, `professionally_pytorch_tutorials`, `professionally_scikit_learn_docs` | D2L, Hugging Face Course, PyTorch docs/tutorials, scikit-learn docs 相关开源材料 | Stage3/Stage4 技术句、术语稳定、encoder/decoder/tokenizer/checkpoint 等自然语境样本 |
| 其他可选语料 | `sharegpt_zh_en_optional`, `translation2019zh_parallel`, `translation2019zh_wiki_zh_monolingual`, `xnli`, `ai_ml_glossary` | ShareGPT optional pool, translation2019zh, XNLI, local AI/ML glossary | 作为候选池或术语资源使用；单语/问答/NLI/对话数据需要转换、筛选或只做辅助参考 |

### Source Handling Notes

部分数据源不能直接进入训练集，原因和处理方式如下：

- `modelscope_wmt_zh_en`：原始 CSV 很大，包含 `zh,en` 列，但部分中文目标存在明显分词空格，例如词与词之间被空格切开。处理时需要中文去空格、标点归一化、长度过滤和噪声过滤，不能作为主来源无控制采样。
- `hf_blog_translation`、`tensorflow_docs_l10n`、`kubernetes_website_l10n`：文档类数据常有标题、代码块、API 参数、markdown 残片、段落错位和页面导航文本。处理时需要按文件路径/标题/段落对齐，再过滤代码模板和社交 URL 噪声。
- `translation2019zh_wiki_zh_monolingual`、`cc_cedict`、`kaikki_chinese`、`ai_ml_glossary`：这类不是标准英中平行句，不能直接当普通翻译样本使用，只能用于术语、中文表达参考或辅助构造。
- `wmt19_zh_en`、`wmt_news`、`translation2019zh_parallel`：正式新闻/书面语较多，适合 general written 和 logic 候选，但需要控制比例，避免 Stage2/Stage4 被新闻、政治、法律、外交语体主导。
- `opus_ccmatrix`、`opus_ccaligned`、`opus_paracrawl`：规模很大但 web-mined 噪声较多，适合做候选池，不适合无过滤全量混入。
- `professionally_*` 技术资料：用于补足 v0 中技术术语完全不稳定的问题，但要过滤代码块、规则说明句和模板化教程残片，保证术语出现在自然完整句中。

## Cleaning and Construction Pipeline

v1 数据构造主要经过以下步骤。

### 1. Source Collection and Flattening

1. 下载或整理公开数据源，保留可直接使用、无需申请/邮件/资格审核的资源。
2. 解压压缩包，去除 GitHub 工程目录、CI 配置、脚本、模型文件、压缩包和无关资源。
3. 将每个数据源扁平化为统一目录结构，每个子目录保留主数据文件和必要 metadata。
4. 对 OPUS Moses 格式、CSV、TSV、JSON、JSONL、Parquet、Markdown 文档等格式分别解析。

### 2. Schema Normalization

所有候选样本最终被统一为：

```json
{"source": "English sentence", "target": "中文译文"}
```

规范化过程包括：

- 统一英译中方向，必要时交换 source / target。
- 将 TSV、CSV、JSON、JSONL、Markdown 文档抽取结果转换为统一 JSONL。
- 去除空 source、空 target、非字符串字段和明显解析失败记录。
- 处理 HTML entity、异常转义、重复空格和中英文标点混杂问题。
- 对中文目标做必要的去空格和标点归一化，尤其是 ModelScope WMT 类分词化中文。

### 3. Language and Alignment Filtering

候选样本需要通过基础质量检查：

- source 必须主要是英文，过滤非英文 source。
- target 必须主要是中文或自然中英混合技术文本，过滤异常中文、乱码和错语种 target。
- 过滤 source / target 明显错位的样本。
- 过滤标题对正文、代码说明对网页导航、API 参数对普通译文等错配样本。
- 过滤纯 URL、社交媒体残片、网页菜单、license、表格碎片和 markdown boilerplate。
- 过滤过短、过长、长度比例异常和标点/数字占比异常的样本。

### 4. Token Length Control

v1 训练脚本使用：

```text
max_src_len = 96
max_tgt_len = 96
```

因此构造数据时要控制 tokenizer 后长度。最终样本需要适配 48k tokenizer，并保证加上 BOS/EOS 后不会超过训练长度限制。

### 5. Duplicate and Leakage Control

构造过程中做多层去重：

- source exact duplicate 去重
- source / target pair exact duplicate 去重
- normalized source overlap 检查
- normalized pair overlap 检查
- train/test 交叉重复检查
- 高频模板 skeleton 检查
- 过度重复 token 和近重复句式检查

Stage5 最终候选还做过定向 index 审计，删除过 non-English source、misalignment、code template noise、social URL noise、abnormal Chinese、strong domain 和 duplicate/near-duplicate 等问题样本。

### 6. Stage-Specific Selection

清洗后的候选池不会一次性混合训练，而是按课程目标进入不同 stage：

- Stage1 优先简单日常句、动作/物体/时间/地点/人物句。
- Stage2 增加普通书面句、中等日常句、简单逻辑句和基础错误回归。
- Stage3 引入技术句、术语稳定和自然技术语境句，同时保留 Stage1/Stage2 clean replay。
- Stage4 平衡 general written、general logic、complex logic、tech logic 和 mixed regression。
- Stage5 做最终收束，混合 replay、terminology stability、complex logic、anti-forgetting 和 anti-pollution stability。

### 7. Anti-Pollution Rules

技术数据尤其需要防污染。以下内容会被过滤或严格限制：

- “正确译法是”“应翻译为”“术语应保持一致”这类规则说明句。
- generated / synthetic / controlled 等模板来源说明残片。
- tokenizer tokenizer tokenizer 这类重复污染。
- API 参数表、函数签名、代码块、导入语句和教程样板句。
- source-target 文档段落错位。
- Transformer 被误翻为“变压器”的样本。
- ordinary words 被技术语境污染的样本，例如 book、paper、window、door 在普通句中被误导成技术含义。

## Not Included

本次 GitHub 更新不包含：

- 每个 stage 的 `test.jsonl`
- 本地数据构造审计文件
- 原始 `available_data_unified/` 数据目录
- checkpoint
- 训练日志
- 推理日志
- 训练报告和推理报告

这些文件会根据后续训练和复核进展决定是否补充。
