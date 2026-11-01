# AIE1903 期中项目：LiDAR 退化模拟下的城市驾驶安全感知与 HD 地图可靠性评估

## 快速导航

| 内容 | 路径 |
|---|---|
| 执行计划（期中/期末、里程碑） | `guideline.md` |
| 官方课程指南 | `docs/course_guideline_official.md` |
| 项目提案 | `docs/midterm_proposal_zh.md` |
| 初步结果汇总 | `docs/midterm_results_summary_zh.md` |
| 闭环结果 | `docs/closed_loop_results_zh.md` |
| 代码说明 | `docs/README_midterm_eval_zh.md` |
| 核心代码 | `eval/` |
| 闭环链路 | `eval/closed_loop.py`（感知→预测→规划→控制） |
| 单元测试 | `tests/` |
| 脚手架（nuScenes 可视化/栅格化） | `utils/` |
| 指标与图表 | `outputs/eval_results/` |
| 闭环结果 | `outputs/eval_results/closed_loop/` |
| 动态 demo（视频/对比） | `outputs/eval_results/demo/` |

## 运行

```bash
conda activate aie1903_ad
# 退化审计（感知与包络指标）
python -m eval.run --frame-step 1 --seeds 3 --out outputs/eval_results

# 闭环安全审计（世界系三层 → IMM → Frenet/A* → MPPI/纯跟踪 → RSS/MRM）
python -m eval.closed_loop \
  --degradations clean range20 range10 drop75 noise01 \
  --combos frenet+mppi frenet+pp astar+mppi astar+pp \
  --max-seconds 20 --out outputs/eval_results/closed_loop

# scene-1094 第 0 帧自车静止：单独从第 13 帧重跑
python -m eval.closed_loop --scenes scene-1094 --start-frame 13 \
  --degradations clean range20 range10 drop75 noise01 \
  --combos frenet+mppi frenet+pp astar+mppi astar+pp \
  --max-seconds 20 --out outputs/eval_results/closed_loop_1094

# 单场景三段 BEV 视频
python -m eval.closed_loop --demo scene-0061:clean:frenet+mppi \
  --max-seconds 20 --out outputs/eval_results/closed_loop

# 合并 1094 修正版并标注碰撞类型
python -m eval.merge_runs --main outputs/eval_results/closed_loop \
  --sub outputs/eval_results/closed_loop_1094 \
  --out outputs/eval_results/closed_loop/metrics_all_enriched.csv

# 闭环汇总图
python -m eval.visualize_loop \
  --metrics outputs/eval_results/closed_loop/metrics_all_enriched.csv \
  --out outputs/eval_results/closed_loop

# 动态 demo（旧版回放）
python -m eval.demo --scene scene-0103 --degradation range10 \
  --start-distance 29 --inject-distance 12 --target-speed 8 \
  --max-steps 140 --out outputs/eval_results/demo

python -m pytest tests -q
```

## 一句话

在 nuScenes mini 的 4 个真实城市场景上，用 LiDAR 退化模拟量程缩短、点云变稀、测距抖动等极端情况，量化哪类退化最先让安全关键感知失效，并用闭环全栈（占据 → 预测 → 规划 → 控制 → MRM）评估地图先验与降级策略的补偿边界。