# V1 Curriculum Checkpoints

本目录保存 V1 curriculum 当前最终选择的 checkpoint。由于完整 checkpoint 文件约 3.1GB，超过 GitHub LFS 单文件 2GB 限制，因此这里按顺序保存为两个分片文件。

## Checkpoint

| 文件 | 说明 |
|---|---|
| `checkpoint_best.part00.pt` | `checkpoint_best.pt` 的第 1 个分片 |
| `checkpoint_best.part01.pt` | `checkpoint_best.pt` 的第 2 个分片 |

这两个分片合并后得到 `checkpoint_best.pt`。该 checkpoint 对应本轮 V1 课程学习训练的最终选择结果。选择时不仅参考统一验证集 `final_eval.jsonl` 上的 valid loss，也结合了固定样例翻译表现。

## Checkpoint Metadata

从 checkpoint 和推理日志中读取到的关键信息：

| 内容 | 记录 |
|---|---|
| epoch | 99 |
| global_step | 394463 |
| tokenizer | `../tokenizer/en_zh_stage_tokenizer_48000/` |
| vocab size | 48000 |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_heads` | 12 |
| `num_layers` | 10 |
| dropout | 0.05 |
| max_src_len | 96 |

## Git LFS

checkpoint 分片文件需要通过 Git LFS 下载。克隆仓库后如果没有看到真实权重内容，请先安装并启用 Git LFS：

```bash
git lfs install
git lfs pull
```

## Usage

先从仓库根目录合并 checkpoint：

```bash
cat encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.part00.pt \
    encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.part01.pt \
    > encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.pt
```

然后运行 V1 推理脚本：

```bash
python3 encoder_decoder/v1_curriculum/src/infer_encoder_decoder_curriculum.py \
  --checkpoint encoder_decoder/v1_curriculum/checkpoints/checkpoint_best.pt \
  --tokenizer-dir encoder_decoder/v1_curriculum/tokenizer/en_zh_stage_tokenizer_48000 \
  --input-file encoder_decoder/v1_curriculum/logs/v1_inference_cases.txt \
  --batch-size 16
```

当前 checkpoint 与 V1 的 48,000 vocab tokenizer 配套使用，不应和 V0 的 16,000 vocab tokenizer 混用。
