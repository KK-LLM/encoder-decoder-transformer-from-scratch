# Checkpoint and Reproducibility

本文只记录当前脚本实际支持的 checkpoint 与复现能力。

## Checkpoint Save Strategy

训练脚本当前每 4 个 epoch 保存一次 checkpoint：

```text
checkpoint_epoch_{epoch}.pt
```

训练结束后保存：

```text
final_model.pt
```

保存 checkpoint 时，脚本也会保存训练时重新训练出的 tokenizer：

```text
minimal_transformer_en_zh_opus_outputs/tokenizer/
```

## Checkpoint Content

当前 `.pt` 文件只包含：

```python
{
    "epoch": epoch,
    "model_state_dict": cpu_state_dict,
}
```

当前脚本不保存 optimizer 状态，不保存 scheduler 状态，也不保存完整 training config 字典。

## Loading for Inference

推理脚本会：

1. 从 checkpoint 读取 `model_state_dict`。
2. 尝试从 checkpoint metadata 读取配置；如果没有，则使用脚本中的 fallback config。
3. 从权重形状推断 vocab size。
4. 校验 tokenizer vocab size 与 checkpoint vocab size 是否一致。
5. 构建模型并 `load_state_dict(..., strict=True)`。
6. 切换到 eval 模式后执行 greedy decode。

## Current Limitations

- 当前训练脚本不支持自动续训。
- 当前 checkpoint 无法精确恢复 optimizer / scheduler 状态。
- 本轮不提交 `.pt`、`.pth`、`.bin`、`.safetensors` 等权重文件。
- 当前训练仍在进行中，完整结果应在训练完成后按真实日志补充。
