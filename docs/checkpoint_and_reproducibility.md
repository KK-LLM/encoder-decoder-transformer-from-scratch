# Checkpoint 与复现性说明

## Checkpoint 保存

训练脚本每 4 个 epoch 在指定输出目录保存一次 checkpoint：

```
minimal_transformer_en_zh_opus_outputs/
├── checkpoint_epoch_4.pt
├── checkpoint_epoch_8.pt
├── ...
├── final_model.pt
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

每个 `.pt` 文件包含：

```python
{
    "epoch": int,
    "model_state_dict": OrderedDict,  # CPU 上的参数
}
```

## Checkpoint 加载

推理脚本 `load_trained_model()` 流程：

1. `torch.load()` 读取 checkpoint 文件（map_location="cpu"）
2. 提取 `model_state_dict` 和训练配置
3. 从权重的 projection/embedding 层形状自动推断 vocab_size
4. 校验 tokenizer vocab_size 与 checkpoint vocab_size 是否匹配
5. 构建与训练时一致的模型结构
6. `model.load_state_dict(state_dict, strict=True)` 加载参数
7. 模型切换到 eval 模式

## 复现说明

通过以下步骤可完整复现训练：

1. 获取 opus-100 en-zh 数据集（从 HuggingFace Hub 或本仓库 Git LFS 数据）
2. 使用本仓库提供的 tokenizer（`v0_baseline/tokenizer/`）
3. 运行 `train_encoder_decoder_opus.py`（修改 device 和相关路径）
4. 训练完成后使用 `infer_encoder_decoder.py` 进行推理

## 当前限制

- 不支持 checkpoint 自动续训（中断后需从头开始）
- 不保存 optimizer/scheduler 状态（无法精确恢复训练状态）
- checkpoint 不包含 training_config 字典（模型配置通过推理脚本的 fallback_config 或从权重形状反推）
