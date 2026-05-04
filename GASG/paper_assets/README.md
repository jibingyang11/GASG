# Paper Asset Index

This directory collects the final figures, tables and CSV summaries selected for the GASG paper. All quantitative results use PersonDataset test samples at `256 x 576`.

## Final Assets

```text
final_paper/
  figures/
    Fig1_overall_pipeline.png
    Fig2_gasg_architecture.png
    Fig3_rightview_quality.png
    Fig4_gasg_ablation.png
    Fig5_depth_visual.png
    Fig6_depth_error_visual.png
    Fig7_rightview_metrics.png
    Fig8_depth_metrics.png
  tables/
    Table1_rightview_generation.tex
    Table2_pseudostereo_depth.tex
    Table3_gasg_ablation.tex
  data/
    rightview_summary.csv
    pseudostereo_depth_summary.csv
```

The same figures and tables are copied into `paper_mdpi_applied_sciences/figures` and `paper_mdpi_applied_sciences/tables` by:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\figures\generate_paper_assets.py
```

## Main Results

GASG obtains `28.56 dB` PSNR, `0.9687` SSIM and `0.0425` LPIPS for right-view synthesis. With DEFOM-Stereo used as the common downstream backend, `GASG -> DEFOM` obtains `0.2974` AbsRel and `4.8595` RMSE, close to the true-right reference of `0.2958` AbsRel and `4.8371` RMSE.

The fair comparison uses PersonDataset-calibrated DA-V2/MoGe warp baselines and PersonDataset-fine-tuned ZeroStereo/StereoGen.
