# WoodyDA-Net：樱桃细长木质结构分割模型

## 1. 模型定位

本目录保存项目自研的 **WoodyDA-Net（Woody-structure Dual-Attention Network）**。它用于休眠期樱桃图像中主干与枝条的二值前景分割，为后续形态学处理、拓扑路由、`mask_clip v3`、芽点挂接和有向结构图构建提供连续的木质结构载体。

该模型不是 iMED-Lab 的官方 CS-Net，也不应写作“改进 CS-Net”或“官方 CS-Net”。目录名、类名 `CSNet` 和历史权重名中的 `csnet` 仅为项目早期保留的兼容标识；论文和新实验统一使用 **WoodyDA-Net**。

## 2. 网络结构

WoodyDA-Net 是一个四级 U 形编码器—解码器二值分割网络。默认构造方式为：

```python
CSNet(
    in_channels=3,
    n_classes=1,
    use_channel_attention=True,
    use_spatial_attention=True,
)
```

整体数据流如下：

```text
ROI 黑背景 RGB 图像（归一化，1024 x 1024）
  -> 编码器：64 -> 128 -> 256 -> 512 通道
  -> 瓶颈层：1024 通道
  -> 解码器与跳跃连接：512 -> 256 -> 128 -> 64 通道
  -> 1 通道 logits
  -> 外部 sigmoid
  -> 概率阈值 0.5，得到二值木质结构载体
```

每个编码器块和解码器块均由两次 `3 x 3 Conv-BN-ReLU` 组成。编码器输出经最大池化下采样；解码器先经反卷积上采样，再与对应编码器特征拼接。

## 3. 双注意力模块

编码器各尺度、1024 通道瓶颈层和解码器各尺度，都会依次经过通道注意力和空间注意力。

### 通道注意力

对特征图分别进行全局平均池化与全局最大池化；两条描述向量经共享的两层 MLP（压缩比 16）后相加，经 sigmoid 得到通道权重，并对原特征图逐通道重标定。它用于强调对木质结构响应更有区分力的特征类型。

### 空间注意力

对通道维分别求均值图和最大值图，将二者拼接后经 `7 x 7` 卷积和 sigmoid 得到空间权重，并对通道重标定后的特征图逐位置重标定。它用于加强细长、连续的空间支持区域。

本模型的结构性特点是：**在编码、瓶颈和解码的全部尺度重复采用“通道注意力 -> 空间注意力”的顺序组合**。论文中应将其表述为面向细长木质结构载体提取的任务适配组合，而不是宣称单独发明了通道注意力或空间注意力。

## 4. 消融开关

模型源文件：`model.py`。训练器可使用以下四种配置：

| 名称 | `use_channel_attention` | `use_spatial_attention` | 实验用途 |
|---|---:|---:|---|
| WoodyDA-Net（A0） | true | true | 已完成的 ROI 主模型条件 |
| 无注意力（A1） | false | false | 验证双注意力相对同骨干网络的贡献 |
| 仅通道（A2） | true | false | 分离通道特征选择的贡献 |
| 仅空间（A3） | false | true | 分离细长空间支持重标定的贡献 |

候选升级模型 `WoodyCAR-Net` 在仅通道条件上，于 `dec2` 和 `dec1` 两个高分辨率解码尺度增加多尺度残差连续性上下文支路。它不使用原有的乘性空间注意力：并联的深度可分离 `3 x 3` 与空洞率为 2 的 `3 x 3` 卷积先提取局部与稍大范围的曲线上下文，随后经 `1 x 1` 融合，并以零初始化的可学习系数残差加入主特征。其初始行为与仅通道模型一致，只有在训练证明有益时才逐步启用该支路。

四种变体共享编码器宽度、解码器宽度、跳跃连接、损失函数、输入尺寸、数据划分和阈值；消融实验中只允许改变注意力是否启用。

## 5. 训练与评估协议

当前正式消融协议固定为：

- 输入：ROI 黑背景 RGB 图像，`1024 x 1024`；
- 标签：只合并 LabelMe 中的 `Trunk` 与 `Branch` 为二值前景；
- 损失：BCE + soft Dice；
- 优化：AdamW，余弦退火学习率；
- 训练：100 epochs，batch size 4，AMP，seed 42；
- 最优权重选择：验证集 IoU 最大；
- 推理：logits 经 sigmoid 后以 `0.5` 阈值二值化；
- 报告：IoU、Dice、precision、recall、clDice、预测连通域数与悬浮连通域率。

ROI 是整个分割对比共享的输入条件，而不是 WoodyDA-Net 独有的网络模块。已完成的 ROI/全图对比表明 ROI 对 WoodyDA-Net、U-Net 和 DeepLabV3+ 都有帮助；因此 A1-A3 只在 ROI 条件下进行，以集中解释 WoodyDA-Net 内部机制。

## 6. 文件说明

```text
branch_seg_csnet_v2/
├── model.py                   # WoodyDA-Net 主网络；保留 CSNet 类名以兼容已有权重
├── loss.py                    # Dice 与 clDice 损失实现
├── CLOUD_HANDOFF.md           # 云端消融实验打包、路径和运行说明
├── requirements_cloud.txt     # 云端最小 Python 依赖
└── README.md                  # 本中文模型说明
```

统一训练入口：

```text
02_code/03_training/compag_experiments/runners/train_branch_seg_binary.py
```

三份 ROI 消融配置：

```text
binary_cleaned_csnet_no_attention_roi_bcedice.yaml
binary_cleaned_csnet_channel_only_roi_bcedice.yaml
binary_cleaned_csnet_spatial_only_roi_bcedice.yaml
binary_cleaned_woodycar_roi_bcedice.yaml
```

一键运行脚本：

```text
02_code/03_training/compag_experiments/scripts/run_woodyda_attention_ablations.sh
02_code/03_training/compag_experiments/scripts/run_woodycar_context_candidate.sh
```

## 7. 结果解释边界

当前完成的 ROI/全图对比中，WoodyDA-Net 的 ROI 条件 IoU 为 0.835，高于同条件 U-Net 的 0.772 和 DeepLabV3+ 的 0.763；该结果来自 50 个含重复来源图像的增强验证样本，支持受控模型选择，不应表述为树级群体显著性或跨数据集普适优越性。

后续 A1-A3 结果仅在完整实验目录返回、核对配置与指标文件后，才能登记为论文证据。分割掩码支撑的是结构感知与拓扑推理，不能将路由成功率、图有效率或芽点挂接率表述为剪枝准确率或植物学正确性。
