# nuScenes v1.0-mini 数据集速览

## 1. 概况

- nuScenes 官方 mini 子集：10 个约 20 秒场景，404 个关键帧，总计约 3.3 分钟。
- 传感器配置与完整版一致：6 相机 + 1 激光雷达 + 5 毫米波雷达。
- 磁盘约 5.2 GB；适合跑通流程、可视化和失效分析，不适合从头训练大模型。

## 2. 目录

| 路径 | 大小 | 内容 |
|---|---:|---|
| `v1.0-mini/` | 32 MB | 13 张 JSON 元数据表 |
| `samples/` | 706 MB | 关键帧文件（404 帧 × 12 传感器） |
| `sweeps/` | 4.4 GB | 关键帧之间的高帧率原始帧 |
| `maps/` | 5.6 MB | 4 张语义先验地图 PNG（0.1 m/px） |
| `expansion/` | 47 MB | Map Expansion v1.3 车道级高精地图 JSON |

## 3. 元数据表

| 表 | 行数 | 作用 |
|---|---:|---|
| `log` | 8 | 采集行程（车辆、日期、城市） |
| `scene` | 10 | 约 20 秒连续片段 |
| `sample` | 404 | 关键帧（2 Hz） |
| `sample_data` | 31,206 | 每个传感器帧 |
| `ego_pose` | 31,206 | 自车全局位姿，与 `sample_data` 一一对应 |
| `calibrated_sensor` | 120 | 传感器外参 + 相机内参 |
| `sensor` | 12 | 传感器通道定义 |
| `sample_annotation` | 18,538 | 3D 框（仅关键帧） |
| `instance` | 911 | 物体轨迹 |
| `category` / `attribute` / `visibility` | 23 / 8 / 4 | 查表 |
| `map` | 4 | 地图与 log 关联 |

关系链：

```text
log → scene → sample → sample_annotation → instance → category
sample → sample_data → ego_pose / calibrated_sensor → sensor
```

坐标：`ego_pose` 与 `sample_annotation.translation` 为全局米制坐标；`size = [宽, 长, 高]`；`rotation = [w, x, y, z]`；`calibrated_sensor.rotation` 是**传感器 → 自车**。

## 4. 传感器与时间结构

| 传感器 | 数量 | 规格 | 频率 | 关键帧 | sweep |
|---|---:|---|---:|---:|---:|
| 相机 | 6 | 1600×900 JPEG | ~12 Hz | 404/路 | ~1,930/路 |
| LIDAR_TOP | 1 | ~34,700 点/帧，32 线 | 20 Hz | 404 | 3,531 |
| 雷达 | 5 | ~120 点/帧，18 字段 | ~13 Hz | 404/路 | ~2,250/路 |

- **关键帧**：2 Hz（0.5 s），12 路同步，**只有关键帧有 3D 标注**。
- **sweep**：关键帧之间的原始帧，无标注。

## 5. 10 个场景

| 场景 | 地区 | 时段 | 关键帧 |
|---|---|---|---:|
| scene-0061 | 新加坡 One North | 白天 | 39 |
| scene-0103 | 波士顿 Seaport | 白天 | 40 |
| scene-0553 | 波士顿 Seaport | 白天 | 41 |
| scene-0655 | 波士顿 Seaport | 白天 | 41 |
| scene-0757 | 波士顿 Seaport | 白天 | 41 |
| scene-0796 | 新加坡 Queenstown | 白天 | 40 |
| scene-0916 | 新加坡 Queenstown | 白天 | 41 |
| scene-1077 | 新加坡 Holland Village | 夜间 | 41 |
| scene-1094 | 新加坡 Holland Village | 夜间 | 40 |
| scene-1100 | 新加坡 Holland Village | 夜间 | 40 |

## 6. 标注

- 18,538 个 3D 框，911 个实例，平均 45.9 框/关键帧（8–156）。
- 类别：car 41.1%、pedestrian.adult 25.7%、barrier 12.5%、trafficcone 7.4%、truck 3.5%，其余 <3%。
- 可见度：80–100% 占 51%，0–40% 占 30%。
- 框内 LiDAR 点数中位数 4；23% 的框没有 LiDAR 点；68% 的框没有雷达点。

## 7. 示例：scene-0061

**采集**：车辆 `n015`，2018-07-24，新加坡 One North。mini 中仅保留这 1 个场景：39 个关键帧、19.1 秒、4,699 个框、227 个实例。

用到的元数据行数：

| 表 | 行数 |
|---|---:|
| `log` / `scene` | 1 / 1 |
| `sample` | 39 |
| `sample_data` / `ego_pose` | 2,963 / 2,963 |
| `sample_annotation` | 4,699 |
| `instance` | 227 |
| `calibrated_sensor` / `sensor` | 12 / 12 |
| `map` | 1 |

用到的文件：**2,963 个，453.8 MB**。

| 通道组 | 文件数 | 大小 |
|---|---:|---:|
| 6 相机 | 1,318（234 关键帧 + 1,084 sweep） | 190.8 MB |
| LiDAR | 382（39 + 343） | 253.0 MB |
| 5 雷达 | 1,263（195 + 1,068） | 10.2 MB |

首关键帧 `ca9a282c…`：12 个文件（6 JPG + 1 `.pcd.bin` + 5 `.pcd`），自车位姿 `[411.42, 1181.20, 0.0]`，69 个 3D 框。LiDAR 相邻帧间隔 50 ms，每 10 帧一个关键帧。

单实例示例：一名行人（`instance 6dd2cbf4…`）在 39 个关键帧中连续出现 39 次，形成一条完整轨迹。

## 8. 格式注意

- LiDAR 是无头二进制 `.pcd.bin`：`np.fromfile(path, np.float32).reshape(-1, 5)`，列为 `x, y, z, intensity, ring`。
- 雷达是标准 PCD（ASCII 头 + binary），18 个字段。
- 相机内参在 `calibrated_sensor.camera_intrinsic`（1600×900，fx≈1266）。
- 环境未安装官方 devkit，直接读 JSON 即可。
- 地图 PNG 为黑白二值图，0.1 m/px：白色=可行驶区域。像素与全局坐标的对应关系为 `col = x/0.1`、`row = H-1-y/0.1`（图像上下翻转）。
- 矢量 HD 地图 `expansion/<location>.json` 是数字化的几何与拓扑：`node` 存 `(x, y)` 坐标，`polygon` 引用节点 token 围成面，`lane`/`road_segment`/`ped_crossing`/`stop_line` 引用 polygon，`traffic_light` 引用 line，`connectivity` 和 `arcline_path_3` 提供车道连通关系与中心线弧段。

## 9. 可视化

```bash
python utils/visualize_nuscenes_mini_scene.py              # 默认 scene-0061
python utils/visualize_nuscenes_mini_scene.py --scene scene-1094
python utils/visualize_nuscenes_mini_scene.py --all        # 全部 10 个 scene
```

每个 scene 输出到 `outputs/datasets_visualize_results/<scene>/`：

- `<scene>_dashboard.png`：6 路相机（含 3D 框投影）+ LiDAR BEV + 自车轨迹 + 统计
- `<scene>_cam_front_boxes.png`：前视相机高清 3D 框
- `<scene>_bev_sequence.png`：全部关键帧的 BEV 序列
- `<scene>_bev.gif`：BEV 动画
- `<scene>_map_trajectory.png`：栅格可行驶地图（局部裁剪）+ 自车轨迹
- `<scene>_vector_map_trajectory.png`：矢量 HD 地图（车道、路口、人行横道、停止线、红绿灯）+ 自车轨迹

`--all` 还会在结果根目录生成：

- `maps_overview.png`：4 张栅格地图全景 + 对应 scene 轨迹
- `vector_maps_overview.png`：4 个区域的矢量 HD 地图全景 + 全部 scene 轨迹

矢量 JSON 也可以栅格化成任意分辨率的 mask：

```bash
python utils/rasterize_hd_map.py --scene scene-0061 --resolution 0.2
```

输出到 `outputs/datasets_visualize_results/map_rasterization/`：8 层 mask（`npz`）、栅格预览图、与官方 PNG 的对比图。`road_segment` 层与官方 PNG 的 IoU≈0.89，说明官方 PNG 就是矢量路面的栅格化产物。

