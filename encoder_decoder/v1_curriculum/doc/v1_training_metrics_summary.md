# V1 课程学习训练记录与指标分析

## 原始日志索引

- `<train_log_dir>/encoder_decoder_stage1_train.log`
- `<train_log_dir>/encoder_decoder_stage2_train.log`
- `<train_log_dir>/encoder_decoder_stage3_train.log`
- `<train_log_dir>/encoder_decoder_stage4_train.log`
- `<train_log_dir>/encoder_decoder_stage5_low_lr_epochs49_64.log`
- `<train_log_dir>/encoder_decoder_stage5_low_lr_epochs65_80.log`
- `<train_log_dir>/encoder_decoder_stage5_low_lr_epochs81_99.log`
- `<train_log_old_dir>/encoder_decoder_stage5_old_params_epochs39_52.log`

其中，`stage5_low_lr` 三个日志文件是新参数 Stage5 的连续训练记录；`stage5_old_params` 是旧参数对照日志，用于分析 Stage5 调参前后的 loss 曲线变化。

## 统一训练设置

| 配置项 | V1 设置 |
|---|---|
| 模型 | 手写 Encoder-Decoder Transformer |
| 主要结构 | Post-LN，Sinusoidal Positional Encoding，普通 Multi-Head Attention |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_layers` | 10 |
| `num_heads` | 12 |
| tokenizer | 统一 48,000 vocab tokenizer |
| 统一验证集 | `final_eval.jsonl`，15,000 条 |
| max length | source 96，target 96 |
| Stage1-4 主要训练参数 | `dropout=0.08`，`warmup_steps=7000`，`label_smoothing=0.03`，Noam `factor=1.0` |
| Stage5 新参数 | `dropout=0.05`，`warmup_steps=1000`，`label_smoothing=0.01`，Noam `factor=0.02` |
| checkpoint 策略 | 每个阶段从上一阶段 best checkpoint 继续训练；每个阶段按最低 valid loss 保存 best checkpoint |

## 阶段目标与规模

| 阶段 | 训练定位 | 训练集规模 | 阶段 test 文件规模（本次不作为验证集） | 统一验证集规模 |
|---|---|---:|---:|---:|
| Stage1 | 基础短句稳定化，建立基础英中对齐 | 107,000 | 2,000 | 15,000 |
| Stage2 | 普通日常句、书面句和简单逻辑扩展 | 281,681 | 6,500 | 15,000 |
| Stage3 | 技术术语和技术语境增强 | 432,987 | 7,444 | 15,000 |
| Stage4 | 复杂逻辑、技术逻辑和 mixed regression | 503,000 | 7,300 | 15,000 |
| Stage5 | 最终综合收束，兼顾 replay、术语稳定、anti-pollution 和 complex_logic | 548,084 | 9,444 | 15,000 |

## 阶段总览

| 阶段 | 日志覆盖 epoch | 训练起点 | 最佳 epoch | 最低 valid loss | 最后一轮 valid loss | 备注 |
|---|---|---|---:|---:|---:|---|
| Stage1 | 1-18 | 随机初始化 | 14 | 4.5573 | 4.6164 | 基础短句开始形成可读输出。 |
| Stage2 | 15-30 | Stage1 best checkpoint | 25 | 3.9273 | 3.9529 | 日常句和简单逻辑能力扩展。 |
| Stage3 | 26-33 | Stage2 best checkpoint | 30 | 3.3192 | 3.3424 | 技术术语开始稳定进入训练目标。 |
| Stage4 | 31-48 | Stage3 best checkpoint | 38 | 3.1282 | 3.1408 | 技术逻辑和复杂逻辑继续增强。 |
| Stage5 | 49-99 | Stage4 checkpoint_epoch_48 | 99 | 2.6998 | 2.6998 | 低学习率最终收束，后期下降幅度变小。 |

## 首轮 / 最佳 / 最终趋势表

| 阶段 | 首轮 train loss | 首轮 valid loss | 最佳 epoch | 最低 valid loss | 最后一轮 train loss | 最后一轮 valid loss | 走势判断 |
|---|---:|---:|---:|---:|---:|---:|---|
| Stage1 | 7.9739 | 6.5969 | 14 | 4.5573 | 3.1352 | 4.6164 | valid loss 先快速下降，后期开始反弹，应使用 best checkpoint。 |
| Stage2 | 3.9906 | 4.1863 | 25 | 3.9273 | 2.9776 | 3.9529 | 日常句扩展有效，但后期仍需要按 best checkpoint 传递。 |
| Stage3 | 1.6301 | 3.3427 | 30 | 3.3192 | 1.3501 | 3.3424 | 技术数据进入后 train loss 较低，valid loss 仍以 best checkpoint 为准。 |
| Stage4 | 2.8566 | 3.1732 | 38 | 3.1282 | 2.3976 | 3.1408 | valid loss 降到 3.1282 后进入平台和小幅波动。 |
| Stage5 | 3.1373 | 2.7900 | 99 | 2.6998 | 2.8501 | 2.6998 | 低学习率连续收束，后期收益变小。 |

## 起始 checkpoint 与最佳 checkpoint

| 阶段 | 训练起点 checkpoint | 最佳 checkpoint | 说明 |
|---|---|---|---|
| Stage1 | 随机初始化 | `<training_output_dir>/stage1_outputs/best_checkpoint.pt`（epoch 14，valid loss 4.5573） | 从零开始建立基础对齐。 |
| Stage2 | `<training_output_dir>/stage1_outputs/best_checkpoint.pt`（last_epoch=14，global_step=8918） | `<training_output_dir>/stage2_outputs/best_checkpoint.pt`（epoch 25，valid loss 3.9273） | 从 Stage1 best checkpoint 继续。 |
| Stage3 | `<training_output_dir>/stage2_outputs/best_checkpoint.pt`（last_epoch=25，global_step=33129） | `<training_output_dir>/stage3_outputs/best_checkpoint.pt`（epoch 30，valid loss 3.3192） | 从 Stage2 best checkpoint 继续。 |
| Stage4 | `<training_output_dir>/stage3_outputs/best_checkpoint.pt`（last_epoch=30，global_step=51794） | `<training_output_dir>/stage4_outputs/best_checkpoint.pt`（epoch 38，valid loss 3.1282） | 从 Stage3 best checkpoint 继续。 |
| Stage5 | `<training_output_dir>/stage4_outputs/checkpoint_epoch_48.pt`（last_epoch=48，global_step=135638） | `<training_output_dir>/stage5_outputs/best_checkpoint.pt`（epoch 99，valid loss 2.6998） | 新参数 Stage5 从 Stage4 checkpoint_epoch_48 继续；旧参数对照从 Stage4 best checkpoint 开始。 |

## 阶段能力观察

- Stage1：主要完成基础短句对齐，固定基础样例已经开始形成可读输出。
- Stage2：普通日常句和简单逻辑能力增强，valid loss 从 4.1863 降到 3.9273。
- Stage3：技术术语进入训练目标，`tokenizer`、`encoder`、`decoder`、`checkpoint` 等术语开始被模型更稳定地处理。
- Stage4：技术逻辑、complex_logic 和 mixed regression 开始混合训练，但复杂句 source following 仍有波动。
- Stage5：低学习率收束后，基础句、技术术语和部分逻辑句稳定性进一步改善，最终 valid loss 降到 2.6998。

## Stage5 参数试验指标

| 训练版本 | 对应日志 | 参数设置 | Epoch 范围 | 首轮 train loss | 首轮 valid loss | 最佳 epoch | 最低 valid loss | 最后一轮 train loss | 最后一轮 valid loss | LR 范围 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 旧参数对照 | `encoder_decoder_stage5_old_params_epochs39_52.log` | `dropout=0.08`，`warmup_steps=7000`，`label_smoothing=0.03`，Noam `factor=1.0` | 39-52 | 3.3326 | 3.0179 | 40 | 3.0089 | 2.9358 | 3.0771 | 0.00011869 -> 0.00009087 |
| 新参数低学习率 | `encoder_decoder_stage5_low_lr_epochs49_64.log` + `encoder_decoder_stage5_low_lr_epochs65_80.log` + `encoder_decoder_stage5_low_lr_epochs81_99.log` | `dropout=0.05`，`warmup_steps=1000`，`label_smoothing=0.01`，Noam `factor=0.02` | 49-99 | 3.1373 | 2.7900 | 99 | 2.6998 | 2.8501 | 2.6998 | 0.00000192 -> 0.00000115 |

## Stage5 调参结论

旧参数下，train loss 从 `3.3326` 降到 `2.9358`，但 valid loss 从最低 `3.0089` 反弹到 `3.0771`，说明训练集继续下降并没有转化成统一验证集收益。新参数通过降低 Noam `factor`、缩短 warmup、降低 label smoothing 和 dropout，把 Stage5 从“大阶段继续训练”调整为“低学习率最终收束”。新参数下 valid loss 从 `2.7900` 持续下降到 `2.6998`，训练曲线更符合最终阶段的目标。

## Final best epoch 固定样例

| 样例类别 | 英文输入 | 模型输出 |
|---|---|---|
| `basic` | the girl opened the window yesterday. | 女 孩 昨 天 打 开 了 这 扇 窗 户 。 |
| `general_logic` | Although some progress has been made, much remains to be done. | 尽 管 取 得 了 进 展 ， 但 仍 有 许 多 工 作 要 做 。 |
| `technical_terms` | The encoder output gives the decoder source-side context in sequence-to-sequence translation. | 编 码 器 输 出 会 为 解 码 器 提 供 源 端 上 下 文 ， 适 用 于 序 列 到 序 列 翻 译 。 |
| `complex_logic` | Because there was not enough time, the child could not wait in the room. | 因 为 时 间 不 够 ， 那 个 孩 子 无 法 在 房 间 里 等 。 |
| `regression_antipollution` | If the tokenizer produces unstable token sequences, translation quality may drop. In LLM inference, engineers use special tokens to avoid fragmented subwords. | 如 果 分 词 器 生 成 的 token 序 列 不 稳 定 ， 翻 译 质 量 可 能 会 下 降 。 在 语 言 模 型 推 理 中 ， 工 程 师 使 用 特 殊 token 来 避 免 碎 片 化 子 词 |
