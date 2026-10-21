# AIE1903 期中项目：LiDAR 退化下的安全感知审计

## 快速导航

| 内容 | 路径 |
|---|---|
| 执行计划（期中/期末、里程碑） | `guideline.md` |
| 官方课程指南 | `docs/course_guideline_official.md` |
| 项目提案 | `docs/midterm_proposal_zh.md` |
| 初步结果汇总 | `docs/midterm_results_summary_zh.md` |
| 代码说明 | `docs/README_midterm_audit_zh.md` |
| 核心代码 | `audit/` |
| 单元测试 | `tests/` |
| 脚手架（nuScenes 可视化/栅格化） | `utils/` |
| 指标与图表 | `outputs/audit_results/` |
| 动态 demo（视频/对比） | `outputs/audit_results/demo/` |

## 运行

```bash
conda activate aie1903_ad
python -m audit.run --frame-step 1 --seeds 3 --out outputs/audit_results
python -m audit.demo --scene scene-0103 --degradation range10 \
  --start-distance 29 --inject-distance 12 --target-speed 8 \
  --max-steps 140 --out outputs/audit_results/demo
python -m pytest tests -q
```

## 一句话

在 nuScenes mini 的 4 个真实场景上，量化距离截断、丢点、噪声与昼夜条件如何侵蚀规划所依赖的安全感知，并审计 HD 地图先验的补偿边界。
* 真实 LiDAR 退化如何影响安全感知