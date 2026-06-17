# Logs

v0_baseline 训练和推理阶段的量化数据，包括训练指标、原始日志和逐条推理结果。

## Files

| 文件 | 说明 |
|---|---|
| `train_metrics.csv` | 48 epoch 完整训练指标（train_loss / valid_loss / lr / checkpoint / best_so_far） |
| `train_log_summary.md` | 训练摘要（Markdown），包含配置、数据、tokenizer、loss 趋势和 best checkpoint 分析 |
| `opus_train_log.log` | 完整训练终端日志，包含每个 epoch 的 loss、lr 和 12 条固定例句翻译输出 |
| `inference_eval_epoch20.jsonl` | 250 条 train / validation / test 抽样推理结果（JSONL），逐条含 source / reference / prediction / quality |
| `fixed_eval_epoch20.jsonl` | 55 条 fixed_simple / fixed_terms / fixed_logic 固定测试集推理结果（JSONL） |
| `inference_metrics_epoch20.csv` | 按 split 汇总的推理指标（长度比、空输出率、重复率等） |
| `tokenizer_term_check_epoch20.txt` | 17 个 ML/CS 术语在训练后 tokenizer 上的实际切分结果 |
