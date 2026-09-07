# mask_topology_routing

纯几何的樱桃树骨架拓扑恢复模块。

## 定位

- 输入：单张 Combined Mask
- 输出：主干 trunk + 侧枝 branch 的矢量拓扑
- 方法：distance transform + skeletonize + Dijkstra route + 图压缩
- 性质：**不涉及模型训练**

## 主要脚本

- `utils.py`：后处理核心逻辑、可视化、结果导出
- `evaluate_visualize.py`：批量评估、指标汇总、6 面板诊断图
- `data.py`：复用的 manifest / 路径辅助工具

## 默认输出

- 结果目录：`04_results/mask_topology_routing/`
- 评估目录：`04_results/mask_topology_routing/evaluation/`

## 启动示例

```bash
D:\app\anna\envs\cherry\python.exe evaluate_visualize.py --num-samples 100
```
