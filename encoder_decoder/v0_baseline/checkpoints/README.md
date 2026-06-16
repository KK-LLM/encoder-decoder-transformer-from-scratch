# Checkpoints

## Checkpoint 保存机制

训练脚本支持以下 checkpoint 保存策略：

- 每 4 个 epoch 保存一次（`epoch % 4 == 0`），文件名格式：`checkpoint_epoch_{epoch}.pt`
- 训练结束后保存 final model：`final_model.pt`
- 同时保存训练时的 tokenizer 到 `tokenizer/` 子目录

每个 checkpoint 文件包含：

```python
{
    "epoch": int,
    "model_state_dict": dict,  # CPU 上的 state_dict
}
```

## 为什么本轮不提交权重文件

1. 当前训练仍在进行中，尚未产生最终模型
2. Checkpoint 文件较大（本项目 model size 约 300-800 MB），不适合直接提交到 GitHub
3. 权重文件已通过 `.gitignore` 排除（`*.pt`）

## 如何加载 Checkpoint

推理脚本中的 `load_trained_model()` 函数演示了完整的 checkpoint 加载流程：

1. `torch.load()` 读取 `.pt` 文件
2. 从 checkpoint 中提取 `model_state_dict` 和训练配置
3. 自动推断 vocab_size 并校验 tokenizer 匹配
4. 构建模型 → `load_state_dict(strict=True)` → 切换到 eval 模式

## 复现说明

通过公开的 opus-100 数据集和本目录中的训练脚本及 tokenizer，可完整复现训练流程。Checkpoint 文件将在本地训练时自动生成。
