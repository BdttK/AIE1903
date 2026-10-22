# AIE1903 期中项目：LiDAR 退化模拟下的城市驾驶安全感知与 HD 地图可靠性评估

## 快速导航

| 内容 | 路径 |
|---|---|
| 执行计划（期中/期末、里程碑） | `guideline.md` |
| 官方课程指南 | `docs/course_guideline_official.md` |
| 项目提案 | `docs/midterm_proposal_zh.md` |
| 初步结果汇总 | `docs/midterm_results_summary_zh.md` |
| 代码说明 | `docs/README_midterm_eval_zh.md` |
| 核心代码 | `eval/` |
| 单元测试 | `tests/` |
| 脚手架（nuScenes 可视化/栅格化） | `utils/` |
| 指标与图表 | `outputs/eval_results/` |
| 动态 demo（视频/对比） | `outputs/eval_results/demo/` |

## 运行

```bash
conda activate aie1903_ad
python -m eval.run --frame-step 1 --seeds 3 --out outputs/eval_results
python -m eval.demo --scene scene-0103 --degradation range10 \
  --start-distance 29 --inject-distance 12 --target-speed 8 \
  --max-steps 140 --out outputs/eval_results/demo
python -m pytest tests -q
```

## 一句话

在 nuScenes mini 的 4 个真实城市场景上，用 LiDAR 退化模拟量程缩短、点云变稀、测距抖动等极端情况，量化哪类退化最先让安全关键感知失效，并评估 HD 地图先验的补偿边界。