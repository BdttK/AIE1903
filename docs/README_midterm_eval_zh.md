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

# 闭环安全审计（4 场景 × 5 退化 × 4 组合）
python -m eval.closed_loop \
  --degradations clean range20 range10 drop75 noise01 \
  --combos frenet+mppi frenet+pp astar+mppi astar+pp \
  --max-seconds 20 --out outputs/eval_results/closed_loop

# 单场景三段 BEV 视频（预测 / 规划 / 执行）
python -m eval.closed_loop --demo scene-0061:clean:frenet+mppi \
  --max-seconds 20 --out outputs/eval_results/closed_loop

python -m eval.closed_loop --scenes scene-1094 --start-frame 13 \
  --degradations clean range20 range10 drop75 noise01 \
  --combos frenet+mppi frenet+pp astar+mppi astar+pp \
  --max-seconds 20 --out outputs/eval_results/closed_loop_1094

# 盲信感知基线（规划与监控忽略 unknown）
python -m eval.closed_loop --safety blind --combos frenet+mppi \
  --degradations clean range10 --out outputs/eval_results/closed_loop_blind

# 静态目标注入（路线前方 25 m）
python -m eval.closed_loop --safety aware blind --combos frenet+mppi \
  --degradations clean range20 range10 --inject-distance 25 \
  --out outputs/eval_results/closed_loop_inject

# 合并 1094 修正版并标注碰撞类型
python -m eval.merge_runs --main outputs/eval_results/closed_loop \
  --sub outputs/eval_results/closed_loop_1094 \
  --out outputs/eval_results/closed_loop/metrics_all_enriched.csv

# 闭环汇总图（成功率 / 碰撞 / MRM / 安全-效率权衡）
python -m eval.visualize_loop \
  --metrics outputs/eval_results/closed_loop/metrics_all_enriched.csv \
  --out outputs/eval_results/closed_loop

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
| `eval/sim.py` | 运动学/动力学自行车（线性轮胎 + 摩擦圆）、Frenet 投影、五次多项式 |
| `eval/world.py` | 世界系三层占据（自车体过滤）、地图、sweep 回放、目标轨迹 |
| `eval/predict.py` | CV + 协调转弯 IMM，多模态轨迹与不确定度 |
| `eval/planning.py` | Frenet 采样规划（风险代价）、风险感知 A* 基线 |
| `eval/control.py` | MPPI 采样控制、纯跟踪 + 速度控制基线 |
| `eval/safety.py` | RSS/TTC 前向扫描、CRUISE/CAUTION/MRM 状态机 |
| `eval/closed_loop.py` | 闭环运行器、2×2 消融、指标与 CSV/NPZ |
| `eval/merge_runs.py` | 合并分场景重跑、按首碰速度标注行驶中/停车碰撞 |
| `eval/visualize_loop.py` | 预测/规划/执行三段 BEV 视频、总览图与组合对比图 |
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
| `closed_loop/metrics.csv` | 场景 × 退化 × planner × controller 的闭环指标 |
| `closed_loop/metrics_all_enriched.csv` | 含 1094 修正帧与行驶中/停车碰撞分类 |
| `closed_loop/steps/*.npz` | 逐步状态、速度、模式、间距、TTC、规划路径 |
| `closed_loop/demo/<run>/closed_loop.mp4` | 预测 / 规划 / 执行三段 BEV |
| `closed_loop/demo/<run>/overview.png` | 轨迹、速度与模式、前向可见性、最近交互 |
| `closed_loop/closed_loop_comparison.png` | 组合 × 退化的成功率、碰撞、MRM 与安全-效率散点 |
| `closed_loop_inject/metrics.csv` | 3 场景 × 3 退化 × aware/blind 静态注入 |
| `closed_loop_inject/closed_loop_comparison.png` | 注入实验的进度/碰撞/MRM/权衡 |

## 指标口径

- **安全包络**：沿自车未来 3 s 路径，纵向 ≤40 m、横向 ≤10 m；冲突子集横向 ≤3.5 m。
- **包络召回**：包络内 GT 目标足迹至少 1 格且被占据栅格覆盖 ≥5% 的比例。
- **false-free rate**：真值障碍格中被判 free 的比例（越低调越“安静”，但可能意味着漏检）。
- **A/B/C**：A=地图 free 但感知 occupied；B=地图 non-free 但感知 free（限路缘 3 m 内）；C=地图 free 但感知 unknown。
- **危险注入**：在自车前方 10–25 m 注入虚拟行人点簇，比较感知规划、地图规划、Oracle 的碰撞。

## 闭环指标

- **世界系三层**：`free / occupied_low / unknown`，其中 `occupied_low` 只保留 0.25–2.5 m 高度的点，避免树冠/建筑外挑误封路面；自车引擎盖点被显式剔除。
- **预测**：IMM（CV + 协调转弯），3 s 视界、0.2 s 步长；报告多模态概率与预测 RMSE。
- **规划**：Frenet 五次多项式候选（横向 5 档 × 速度 4 档），代价含碰撞、unknown、舒适度、进度与预测目标；风险感知 A* 为基线。
- **控制**：MPPI（K=192、2.5 s）跟踪规划；纯跟踪 + 速度控制为基线；被控对象为线性轮胎 + 摩擦圆的单轨动力学模型。
- **安全**：沿自车当前横向偏移的路线走廊扫描 free/unknown，计算 RSS 安全距离与 TTC；CRUISE → CAUTION → MRM 带 1.5 s 恢复迟滞。前向 50 m 走廊内 unknown 比例 ≥0.65 视为“盲区过大”，直接进入 MRM（range10 因此会长时间停车）；`--safety blind` 让规划器与监控都把 unknown 当 free，用作“盲信感知”的失败基线。
- **闭环指标**：成功率、碰撞（区分行驶中碰撞 / 停车被撞）、最小间距、最小 TTC、到达时间、行驶距离、平均速度、MRM 次数/时长、最大横向加速度、cross-track RMS、预测 RMSE、平均 unknown 比例。
- **静态注入**：在路线前方 `--inject-distance` 处放一个 1.2×1.2×1.5 m 的合成障碍，点簇同样经过退化（距离截断/丢点/噪声）；用于观察 aware 与 blind 栈在最小间距上的差异。

## 已知局限

- 其他交通参与者按日志轨迹开环回放，不会对自车行为做出反应；因此会出现“行人走进已停车的自车”的伪碰撞。指标把自车速度 ≥0.8 m/s 的碰撞记为行驶中碰撞，<0.8 m/s 的记为停车被撞，两者分开报告。
- 碰撞只统计与标注目标（GT box）的 OBB 重叠；未标注的路侧静态结构不进入碰撞计数，只通过占据层的制动体现。
- 传感器回放使用空间上最近的 sweep，并按该 sweep 的真实位姿变换到世界系；自车大幅偏离真实路径时，可见性只是近似。
- 闭环结论依赖安全阈值（RSS 参数、unknown 阈值、MRM 减速度），文中报告参数并做敏感性说明。
- 真实关键帧的路径冲突稀疏，主统计仍用包络指标；闭环实验是补充证据，不替代统计结论。
