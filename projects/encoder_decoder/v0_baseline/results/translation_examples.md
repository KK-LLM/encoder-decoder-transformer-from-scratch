# Translation Examples Across Training

## Purpose

本文件记录训练过程中固定例句在不同 epoch 的 greedy decoding 输出，用于人工观察模型翻译行为的变化趋势。

**注意**：这不是正式评测集结果，不能替代 BLEU / chrF / sacreBLEU 等自动评测。所有输出均为模型实际生成，未做任何润色或修正。

---

## Selected Epochs

| Epoch | Reason |
|---|---|
| 4 | Early checkpoint，模型刚开始学习部分短句模式 |
| 12 | Mid training，valid loss 已明显下降 |
| 20 | Best valid loss，对应推荐 checkpoint |
| 32 | Later training，过拟合趋势开始显现 |
| 48 | Final epoch，展示训练结束时的模型状态 |

---

## Epoch 4 - Early Checkpoint

**train_loss=3.8329, valid_loss=3.7286, lr=0.000523**

| English Input | Model Output (ZH) | Observation |
|---|---|---|
| I need a cup of coffee. | 我 要 咖 啡 | 短句翻译接近正确，缺少量词"一杯" |
| The meeting starts at nine tomorrow. | 明 天 早 上 开 始 | 大致可读，缺少时间"9 点"和主语"会议" |
| This is a small Transformer model. | 这 是 一 个 新 的 电 子 版 本 。 | 语义完全偏移，"Transformer"未被识别 |
| We need more clean data. | 我 们 需 要 一 些 技 术 。 | "data"翻译为"技术"，概念混淆 |
| I'll be there in about ten minutes, just wait for me. | 我 等 等 等 等 等 等 等 等 等 等 等 等 等 等 等 等 | 明显重复，长句生成失败 |
| To be honest, I'd rather stay home than go out tonight. | 我 们 今 晚 就 会 回 来 的 | 语义部分保留（"今晚"+"回来"），但缺少"宁愿"的语义 |

---

## Epoch 12 - Mid Training

**train_loss=3.4021, valid_loss=3.4938, lr=0.000302**

| English Input | Model Output (ZH) | Observation |
|---|---|---|
| I need a cup of coffee. | 我 要 去 咖 啡 | 短句已有"我要"+"咖啡"结构，动词"去"多余 |
| The meeting starts at nine tomorrow. | 上 午 10 时 开 会 | "会议"隐式出现（"开会"），时间“9 点”变成了“10 时” |
| This is a small Transformer model. | 这 是 一 个 模 型 。 | 成功识别"模型"，但丢失"Transformer"和"small" |
| We need more clean data. | 我 们 需 要 数 据 。 | 语义基本正确，缺少"clean"（干净的） |
| I'll be there in about ten minutes, just wait for me. | 我 等 着 一 个 小 时 | 比 epoch 4 的重复输出好很多，但"ten minutes"被译为"一个小时" |
| To be honest, I'd rather stay home than go out tonight. | 我 今 晚 就 要 去 了 ， 亲 爱 的 ， 我 不 会 再 做 一 个 人 了 | 长句出现语义偏离和幻觉，"stay home"完全丢失 |

---

## Epoch 20 - Best Validation Loss

**train_loss=3.2443, valid_loss=3.4693 (best), lr=0.000234**

| English Input | Model Output (ZH) | Observation |
|---|---|---|
| I need a cup of coffee. | 我 要 喝 咖 啡 | 本轮最佳：包含"我要喝咖啡"，比 epoch 4 多了动词"喝" |
| The meeting starts at nine tomorrow. | 明 天 开 会 。 | 保留了“明天开会”的核心语义，但丢失具体时间“9 点” |
| This is a small Transformer model. | 这 是 一 个 计 算 机 。 | "Transformer model"被译成"计算机"，语义偏移 |
| We need more clean data. | 我 们 需 要 数 据 | 核心语义正确，但"clean"和"more"均丢失 |
| I'll be there in about ten minutes, just wait for me. | 我 等 了 十 分 钟 | "ten minutes"正确翻译为"十分钟"，但"just wait for me"语义丢失 |
| To be honest, I'd rather stay home than go out tonight. | 我 今 晚 就 要 回 家 了 | 语义压缩为"今晚回家"，丢失"宁愿待在家而不是出门"的复杂语义 |

---

## Epoch 32 - Later Training

**train_loss=3.1097, valid_loss=3.5089, lr=0.000185**

| English Input | Model Output (ZH) | Observation |
|---|---|---|
| I need a cup of coffee. | 我 要 喝 咖 啡 | 短句保持稳定，与 epoch 20 一致 |
| The meeting starts at nine tomorrow. | 明 天 是 一 个 明 天 。 | 语义退化，出现无意义的同义重复 |
| This is a small Transformer model. | - 这 是 一 个 超 级 计 算 机 。 | 出现多余的 dash 前缀，"small"被译为"超级"，完全相反 |
| We need more clean data. | 我 们 需 要 一 些 数 据 。 | 与 epoch 12 类似，多了一个"一些" |
| I'll be there in about ten minutes, just wait for me. | 等 等 ， 我 一 分 钟 就 到 了 | "ten minutes"变成"一分钟"，但"等等"+"就到了"整体可读 |
| To be honest, I'd rather stay home than go out tonight. | 今 晚 我 要 回 家 了 ， 不 要 再 担 心 了 | 出现语义附加（"不要再担心了"），模型在"脑补"内容 |

---

## Epoch 48 - Final Epoch

**train_loss=2.9984, valid_loss=3.5653, lr=0.000151**

| English Input | Model Output (ZH) | Observation |
|---|---|---|
| I need a cup of coffee. | 我 要 喝 咖 啡 | 短句仍然正确，短句似乎对过拟合不敏感 |
| The meeting starts at nine tomorrow. | 明 天 开 会 | 恢复了“明天开会”的核心语义，但仍缺少时间“9 点” |
| This is a small Transformer model. | 这 是 一 个 很 有 趣 的 计 算 机 。 | 仍然译为"计算机"，但多了"很有趣的"，语义偏移 |
| We need more clean data. | 我 们 需 要 数 据 | 核心语义正确，但细节丢失 |
| I'll be there in about ten minutes, just wait for me. | 等 等 ， 我 一 小 时 内 就 到 了 | "ten minutes"变成"一小时内"，时间语义部分保留但精度下降 |
| To be honest, I'd rather stay home than go out tonight. | 我 今 晚 不 会 再 有 人 陪 我 去 | 语义严重偏移，"有人陪我去"是完全不同的含义 |

---

## Overall Observation

1. **短句更早学会**："I need a cup of coffee." 从 epoch 4 的"我要咖啡"到 epoch 20 的"我要喝咖啡"，短句在早期就能学到基本语义，且对过拟合不敏感。

2. **常见生活表达比技术句更稳定**："The meeting starts at nine tomorrow." 在 epoch 20 和 epoch 48 能保留“明天开会”的核心语义，但中间 epoch 32 出现退化（输出“明 天 是 一 个 明 天”），说明固定例句表现存在波动；而 "This is a small Transformer model." 从未被正确翻译，总是译为"计算机"或"模型"。

3. **长句容易出现语义压缩、偏移或幻觉**："To be honest, I'd rather stay home than go out tonight." 在所有 epoch 中均未正确表达"宁愿待在家"的核心语义，后期更出现"有人陪我去"等完全偏离的生成。

4. **Best epoch 不等于 perfect model**：epoch 20 的翻译质量在全局中相对最好，但仍然存在大量语义偏差。greedy decoding 的局限在复杂句上尤为明显。

5. **Final epoch 不优于 best checkpoint**：epoch 48 的 valid_loss (3.5653) 远高于 epoch 20 (3.4693)，翻译质量直观上也未能超越 epoch 20。训练结束时模型在训练集上 fit 得更好（train_loss 更低），但泛化能力下降。
