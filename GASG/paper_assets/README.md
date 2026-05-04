# 论文图表素材清单

本目录集中整理论文中建议直接使用的图表。所有实验均来自 PersonDataset test，主要分辨率为 `256 x 576`。

## 表格

| 文件 | 用途 |
|---|---|
| `tables/table_rightview.tex` | 右图生成质量主表：PSNR、SSIM、LPIPS、速度、参数量 |
| `tables/table_depth_models.tex` | 深度估计模型性能主表：AbsRel、SqRel、RMSE、RMSElog、d1/d2/d3、速度 |

## 右图生成图

| 文件 | 建议用途 |
|---|---|
| `figures/rightview/rightview_quality_bars.png` | 右图生成质量柱状图 |
| `figures/rightview/rightview_runtime_params.png` | 右图生成速度和参数量对比 |
| `figures/rightview/rightview_quality_visual_grid.png` | 左图、真实右图、不同方法生成右图的可视化对比 |
| `figures/rightview/rightview_ablation_visual_grid.png` | GASG 消融实验可视化：warp-only、direct-only、no-refine、full |
| `figures/rightview/rightview_qualitative_grid.png` | 右图生成定性补充图 |

## 深度估计图

| 文件 | 建议用途 |
|---|---|
| `figures/depth/depth_metrics_bars.png` | 各深度估计模型指标柱状图 |
| `figures/depth/depth_runtime_bars.png` | 各深度估计模型推理时间对比 |
| `figures/depth/depth_comparison_grid.png` | 输入图、GT 深度、不同模型预测深度可视化 |
| `figures/depth/depth_error_grid.png` | 不同模型深度绝对误差可视化 |

## 原始结果

| 文件 | 用途 |
|---|---|
| `raw_results/rightview_summary.md` | 右图生成指标 Markdown 汇总 |
| `raw_results/rightview_summary.csv` | 右图生成指标 CSV |
| `raw_results/rightview_summary.json` | 右图生成指标 JSON |
| `raw_results/depth_summary.md` | 深度估计指标 Markdown 汇总 |
| `raw_results/depth_summary.csv` | 深度估计指标 CSV |
| `raw_results/depth_summary.json` | 深度估计指标 JSON |

## 论文写作提醒

1. LPIPS 越低越好，GASG full 的 LPIPS 为 `0.0425`，优于 `0.10-0.20` 的常见区间。
2. DA-V2、MoGe、Metric3D-V2 等单目方法如果标注为 `median aligned`，表示用了 GT 中位数缩放，不能写成真正的单图绝对深度输出。
3. GASG + DEFOM 与 DEFOM real-right 上界非常接近，可作为“单摄像机伪双目接近真实双目输入”的核心论据。
4. ZeroStereo/StereoGen 使用官方预训练权重，但在 PersonDataset 上右图质量和深度结果都明显低于 GASG，适合作为生成式 baseline 对比。
