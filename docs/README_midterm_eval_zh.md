# 期中项目代码说明（LiDAR 退化安全感知评估）

## 环境与数据

```bash
conda activate aie1903_ad
# 数据：data/nuscenes_full/v1.0-mini + samples/ + maps/ + expansion/
```

## 运行

```bash
# 全量（4 场景、160 关键帧、11 档退化、3 seeds）
python -m eval.run --frame-step 1 --seeds 3 --out outputs/eval_results

# 快速冒烟（约 1 分钟）
python -m eval.run --frame-step 8 \
  --degradations clean range20 range10 drop50 --seeds 2 \
  --inject-frames 0 --inject-distances 10 20 \
  --inject-degradations clean range20 range10 \
  --out /tmp/eval_quick

# 动态 demo（scene-0103, range10, 虚拟行人）
python -m eval.demo --scene scene-0103 --degradation range10 \
  --start-distance 29 --inject-distance 12 --target-speed 8 \
  --max-steps 140 --out outputs/eval_results/demo

# 单元测试
python -m pytest tests -q
```

## 模块

| 文件 | 作用 |
|---|---|
| `eval/core.py` | 网格、坐标变换、占据/自由空间、真值栅格化、退化、包络指标、A*、OBB |
| `eval/experiments.py` | 退化矩阵与逐帧指标、汇总 CSV |
| `eval/map_consistency.py` | 地图-感知差异 A/B/C、归因、全局持续性图 |
| `eval/inject.py` | 虚拟目标注入 + 感知/地图/Oracle 重规划对比 |
| `eval/visualize.py` | 退化曲线、昼夜对比、距离/类别剖面、失效画廊 |
| `eval/policy.py` | 前向走廊扫描、perception/map/degradation-aware 速度策略 |
| `eval/demo.py` | 动态回放、虚拟行人注入、三策略 4 联视频 |
| `eval/run.py` | 一条命令跑全流程 |
| `eval/smoke.py` | 单帧四联图冒烟测试（开发用） |

## 输出（`outputs/eval_results/`）

| 路径 | 内容 |
|---|---|
| `metrics/per_frame.csv` | 每帧 × 退化 × seed 的感知与包络指标 |
| `metrics/hazards.csv` | 每个包络目标的距离、TTC、覆盖率、是否检出 |
| `metrics/summary_by_scene.csv` | 场景 × 退化汇总（含 95% CI） |
| `metrics/summary_by_degradation.csv` | 昼夜 × 退化汇总 |
| `metrics/map_metrics.csv` | 地图差异 A/B/C、归因、自车暴露 |
| `metrics/injection_metrics.csv` | 注入距离 × 退化下的检出与碰撞 |
| `metrics/map_global_<scene>.npz` | 全局差异持续性栅格 |
| `figures/fig_degradation_curves.png` | 包络召回与 false-free 随退化变化 |
| `figures/fig_day_night.png` | 昼夜对比 |
| `figures/fig_distance_profile.png` | 检出率/覆盖率随纵向距离变化 |
| `figures/fig_category_profile.png` | 按类别的检出率 |
| `figures/fig_tradeoff.png` | 召回 vs false-free 权衡 |
| `figures/failure_<scene>.png` | 失效画廊 |
| `figures/map_discrepancy_<scene>.png` | 地图差异三联图 |
| `figures/map_persistence_<scene>.png` | A/B 差异全局热度 |
| `figures/injection_<scene>.png` | 注入目标四联图 |
| `figures/injection_summary.png` | 注入距离-检出/碰撞曲线 |
| `demo/demo_<scene>_<deg>_<policy>.mp4` | 三策略动态 4 联视频 |
| `demo/comparison_<scene>_<deg>.png` | 速度/自由距离/间距对比 |
| `demo/metrics_<scene>_<deg>.csv` | 碰撞、最小间距、停车比例 |

## 指标口径

- **安全包络**：沿自车未来 3 s 路径，纵向 ≤40 m、横向 ≤10 m；冲突子集横向 ≤3.5 m。
- **包络召回**：包络内 GT 目标足迹至少 1 格且被占据栅格覆盖 ≥5% 的比例。
- **false-free rate**：真值障碍格中被判 free 的比例（越低调越“安静”，但可能意味着漏检）。
- **A/B/C**：A=地图 free 但感知 occupied；B=地图 non-free 但感知 free（限路缘 3 m 内）；C=地图 free 但感知 unknown。
- **危险注入**：在自车前方 10–25 m 注入虚拟行人点簇，比较感知规划、地图规划、Oracle 的碰撞。

## 已知局限

- 真实关键帧的路径冲突稀疏，主统计用包络指标；碰撞结论仅来自明确标注的注入实验。
- 自由空间用角度可见性剖面，稀疏点云会产生 unknown 楔形。
- 地图“可行驶”为二值语义 PNG，未区分人行道/停车区；B 类已限制在路缘 3 m 内。
- 单帧静态评估，不含动态预测误差；final 可加 3 s 动态推演。
