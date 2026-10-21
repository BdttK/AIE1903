# AIE1903 — 示例项目选题

这些是起点，不是要求。你们小组可以组合这些想法、修改它们，或提出新的东西——只要它明确与自动驾驶 AI 相关，并且在课程约束内可行（不需要 GPU，不需要大型数据集，不需要实体硬件）。

难度：★ = 基础 · ★★ = 中等 · ★★★ = 进阶（强组可选）

---

### 1. 车道线检测方法对比（★）
在课程示例图像上比较两种车道线检测方法（例如霍夫变换 vs. 颜色阈值滑动窗口）。报告准确率、速度和失效模式。
- 工具：OpenCV、NumPy、Matplotlib
- 数据：`data/sample_images/`、`data/sample_lanes/`

### 2. 目标检测失效案例研究（★）
分析一个简单检测器（或预计算检测结果）何时失效：遮挡、低光照、小目标。构建一个简短的失效案例画廊，并解释每次失效发生的原因。
- 工具：OpenCV、已保存的检测结果
- 数据：`data/sample_detections/`、`data/sample_images/`

### 3. 可行驶区域分析（★）
从分割掩码中提取可行驶区域，计算跨帧的可行驶区域占比，并讨论这如何输入到规划中。
- 工具：NumPy、Matplotlib
- 数据：`data/sample_segmentation/`

### 4. BEV 可视化仪表盘（★★）
构建一个仪表盘，对同一场景并排显示相机图像、BEV 高度图、投影 3D 框和占据栅格。
- 工具：Matplotlib（或 Plotly，若稳定）、NumPy
- 数据：`data/sample_bev/`、`data/sample_3d_boxes/`、`data/toy_occupancy/`

### 5. 点云浏览器（★★）
用于示例点云的交互式浏览器：下采样、按高度着色、高亮 3D 框、旋转视角。（使用 Matplotlib 3D；Open3D 可选。）
- 工具：Matplotlib 3D、NumPy
- 数据：`data/sample_pointclouds/`、`data/sample_3d_boxes/`

### 6. 玩具 HD 地图编辑器（★★）
用 JSON 创建你自己的小型 HD 地图（车道、路口、红绿灯），将其可视化，并为地图扩展一个新特征（例如一座环岛）。
- 工具：Python、JSON、Matplotlib、NetworkX
- 数据：`data/toy_hd_map/`

### 7. 路径规划对比（★★）
在课程路径图和栅格地图上比较 Dijkstra 与 A*。测量路径长度、扩展节点数和运行时间。
- 工具：NetworkX、NumPy
- 数据：`data/toy_hd_map/route_graph.json`、`data/toy_grid_maps/`

### 8. 轨迹预测对决（★★）
在示例轨迹上比较恒速模型与一个简单的学习基线（例如对历史做线性回归）。报告 RMSE 随预测时域的变化。
- 工具：scikit-learn、NumPy、Matplotlib
- 数据：`data/sample_trajectories/`

### 9. 预测式刹车模拟器（★★★）
结合预测 + 规划：预测某个交通参与者的运动，检查与自车的碰撞，并在一个简单的时间线可视化中触发“刹车”决策。
- 工具：NumPy、Matplotlib
- 数据：`data/sample_trajectories/`、`data/toy_grid_maps/`

### 10. 路径跟踪参数研究（★★）
用不同的前视距离和速度运行第 8 周的纯跟踪（pure-pursuit）仿真；生成最大横向误差的热力图。讨论调参的权衡。
- 工具：NumPy、Matplotlib
- 数据：`data/toy_control_paths/`

### 11. VLM 驾驶场景分析（★★）
使用视觉语言模型（API 或预存响应）分析驾驶图像：描述场景、识别危险、将 AI 解释与人类判断进行比较。报告优势与失效案例。
- 工具：任意 VLM API（可选）或已保存的示例
- 数据：`data/vlm_examples/`、`data/sample_images/`

### 12. 安全失效案例分析（★）
使用课程框架分析真实的自动驾驶事故（来自新闻/公开报告）：是感知？预测？规划？还是人类监督？为安全的 AI 设计总结教训。
- 工具：无需工具（分析 + 幻灯片）
- 数据：公开报告（教师提供的链接）

### 13. 行业架构对比（★）
比较两种工业界方法（例如基于摄像头的端到端 vs. 多传感器模块化，基于 HD 地图 vs. 无地图）。展示架构图、优势、局限。
- 工具：无需工具（分析 + 幻灯片）
- 数据：公开技术博客/白皮书

### 14. 占据栅格碰撞检查器（★★★）
构建一个碰撞检查器：给定玩具占据栅格和一条候选路径，报告该路径是否无碰撞。与 A* 输出集成。
- 工具：NumPy、Matplotlib
- 数据：`data/toy_occupancy/`、`data/toy_grid_maps/`

### 15. 你自己的想法（难度不定）
任何与自动驾驶相关、安全、可行且解释清楚的想法。在最终确定之前与教师讨论。
