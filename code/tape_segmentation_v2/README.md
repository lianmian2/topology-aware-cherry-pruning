# Tape Segmentation V2

`Tape_Segmentation_V2` 使用去重后的合并标注重新构建条带分割训练集，并将训练输出统一保存在当前目录。

## 当前版本说明

- 数据源使用 `mmdet_data/annotations/Tape_merged_dedup.json` 和 `mmdet_data/tape/`。
- 训练集按 `Tree ID` 进行 `8:1:1` 切分，避免同树数据泄露。
- 模型结构为 `MobileNetV2 + U-Net`。
- 预处理默认使用 `GrabCut` 收紧边界，当前默认参数为：
  - `GRABCUT_ITERATIONS = 5`
  - `VERTICAL_SHRINK = 0`
- `data_processing.py` 运行时会在终端显示进度条。
- `Graphical_Interface/unified_system/main_gui.py` 默认读取本目录下最新的 `outputs/best_model.pth`。

## 目录结构

```text
Tape_Segmentation_V2/
├── data_processing.py
├── preview_refine_params.py
├── train.py
├── visualize.py
├── README.md
├── mmdet_data/
│   ├── img_dir/
│   ├── ann_dir/
│   └── split_report.json
└── outputs/
    ├── best_model.pth
    ├── training_metrics.json
    └── vis/
```

## 使用流程

### 1. 生成训练数据

```bash
cd <repository-root>
python data_processing.py
```

运行后会显示类似下面的进度：

```text
Processing [##########--------------------] 120/356 split=train file=tree_007_before_view_01.jpg
```

### 2. 开始训练

```bash
python train.py
```

训练完成后，最佳权重会保存到：

```text
Tape_Segmentation_V2/outputs/best_model.pth
```

### 3. 测试集可视化

```bash
python visualize.py
```

可视化会输出 `Original / Ground Truth / Prediction / Overlay` 四联图，便于观察预测边界是否过粗。

## 参数预览

如果需要在重新生成数据前先观察收紧效果，可使用交互预览脚本：

```bash
python preview_refine_params.py --image tree_007_before_view_01.jpg
```

说明：

- `gc_iter` 表示 `GrabCut` 迭代次数。
- `v_shrink` 表示仅在垂直方向收紧的强度。
- 当前训练默认值为 `grabcut_iterations=5, vertical_shrink=0`。

## main_gui 同步

`main_gui` 中的条带分割模型由 `Graphical_Interface/unified_system/logic_models.py` 加载，默认读取：

```text
Tape_Segmentation_V2/outputs/best_model.pth
```

因此重新训练后，只要新的最佳权重已经写入 `outputs/best_model.pth`，GUI 会自动使用最新模型。

## 备注

- `Tape_merge_report.json` 已识别并移除了 `12` 张完全重复图。
- 当前推荐直接用 `Tape_merged_dedup.json` 训练，不再单独混用 `Tape.json` 和 `3-15.json`。
