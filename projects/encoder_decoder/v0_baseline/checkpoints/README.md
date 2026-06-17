# Checkpoints

v0_baseline 训练完成后已提交 best checkpoint：`checkpoint_epoch_20.pt`。

## Best Checkpoint

| 属性 | 值 |
|---|---|
| 文件 | `checkpoint_epoch_20.pt` |
| 大小 | 338 MB（Git LFS） |
| epoch | 20 |
| 选择依据 | valid_loss 全局最低（3.4693） |
| 内容 | `epoch` + `model_state_dict` |
| 注意 | 不含 optimizer / scheduler / RNG，不支持完整自动续训 |

## Save Strategy

训练脚本当前保存策略：

- 每 4 个 epoch 保存一次：`checkpoint_epoch_{epoch}.pt`
- 训练结束后保存：`final_model.pt`
- checkpoint 保存时同步保存训练后的 tokenizer 到输出目录

输出目录由训练脚本设置为：

```text
minimal_transformer_en_zh_opus_outputs/
```

## Checkpoint Content

当前 `.pt` 文件只包含：

```python
{
    "epoch": epoch,
    "model_state_dict": cpu_state_dict,
}
```

## Current Limitations

- 当前 checkpoint 不保存 optimizer 状态。
- 当前 checkpoint 不保存 scheduler 状态。
- 当前脚本不支持自动续训。
- checkout 后可加载 model_state_dict 作为继续微调或阶段训练的初始化权重。
