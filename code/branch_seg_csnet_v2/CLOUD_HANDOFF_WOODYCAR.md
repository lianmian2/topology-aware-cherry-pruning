# WoodyCAR-Net 云端训练交接说明

## 任务

训练 **WoodyCAR-Net** 候选网络，并与已完成的“仅通道注意力”结果进行同协议比较。

WoodyCAR-Net 保留通道注意力，取消原乘性空间注意力；只在 `dec2` 和 `dec1` 高分辨率解码阶段加入多尺度残差曲线连续性上下文支路。该支路由深度可分离 `3 x 3`、空洞率为 2 的 `3 x 3` 和 `1 x 1` 融合组成，并以初始值为零的可学习残差系数接入主干。

## 解压后的操作

1. 保持压缩包内的目录层级不变。
2. 打开 `binary_cleaned_woodycar_roi_bcedice.yaml`，只把三项数据路径改成云端实际路径：
   - `data.train_img_dir`
   - `data.val_img_dir`
   - `data.labelme_dir`
3. 不修改模型名、ROI 输入模式、数据划分、图像尺寸、损失、随机种子、阈值和训练轮数。
4. 从解压后的项目根目录执行：

```bash
chmod +x 02_code/03_training/compag_experiments/scripts/run_woodycar_context_candidate.sh
bash 02_code/03_training/compag_experiments/scripts/run_woodycar_context_candidate.sh
```

短连接检查可用：

```bash
EPOCHS_ARG="--epochs 1" bash 02_code/03_training/compag_experiments/scripts/run_woodycar_context_candidate.sh
```

## 固定协议

- 输入：ROI 黑背景 RGB 图像，1024 x 1024；
- 标签：LabelMe 中 `Trunk` 与 `Branch` 的二值并集；
- 损失：BCE + soft Dice；
- 优化：AdamW，余弦退火；
- 训练：100 epochs，batch size 4，AMP，seed 42；
- 最优权重：验证 IoU 最大的 `best_primary.pth`；
- 输出阈值：sigmoid 概率大于 0.5。

## 需要返回的结果

请完整返回本次输出目录，不要重命名或删减：

```text
configs/
checkpoints/
metrics/
logs/
visualizations/
report/
```

重点保留 `metrics/summary.json`、`metrics/per_epoch.csv`、`metrics/per_sample.csv` 与 `checkpoints/best_primary.pth`。结果比较必须同时提供 IoU、Dice、clDice、预测连通域数和悬浮连通域率；单次训练结果仅用于候选筛选，不能直接作为跨种子的稳定结论。
