# GASG Pseudo-Stereo Depth Estimation

GASG 是一个面向伪双目深度估计任务训练的轻量级左图到右图生成模型。核心流程是：

```text
single left image -> GASG -> pseudo right image
left image + pseudo right image -> stereo depth estimator -> absolute depth
```

论文贡献是 **GASG 伪右图生成器及其可插拔的双目接口**，不是某一个固定的 `GASG + DEFOM-Stereo` 组合。DEFOM-Stereo 在本项目中只是用于实验验证的代表性 stereo backend。

## Clean Project Layout

```text
checkpoints/
  gasg_best.pth                         # best GASG checkpoint, do not delete
  depth_anything_v2_large.pth           # DA-V2 warp and ZeroStereo dependency

configs/
  gasg_config.yaml                      # GASG training config
  eval_config.yaml

data/
  PersonDataset/                        # train/test left, right, depth
  zerostereo_person_test/               # official ZeroStereo generated right views
  stereospace_person_test/              # official StereoSpace HF generated samples
  hf_cache/                             # HF cache for MoGe/offline reuse

models/
  gasg_net.py                           # GASG model

scripts/
  train_gasg.py                         # train GASG
  eval_rightview_compare.py             # right-view quality, speed and ablations
  merge_rightview_results.py            # merge right-view JSON results
  eval_person_depth_model.py            # one pseudo-stereo depth method
  run_person_depth_models.py            # run multiple depth methods
  merge_person_depth_results.py         # merge depth JSON results
  baselines/eval_zerostereo_official.py # official ZeroStereo/StereoGen reproduction
  baselines/eval_stereospace_hf.py      # official StereoSpace HF Space reproduction
  figures/create_overall_pipeline_fig.py
  figures/generate_paper_assets.py      # final figures and tables

results/
  rightview_generation/                 # final right-view generation results
  pseudostereo_depth/                   # final pseudo-stereo depth results

paper_assets/final_paper/               # final figures, tables and CSV summaries
paper_mdpi_applied_sciences/            # MDPI LaTeX package
```

Official model names such as `Depth-Anything-V2` are kept unchanged because third-party imports and checkpoint names depend on them.

## Environment

The experiments were run with:

```powershell
D:\miniconda\envs\myenv\python.exe
```

Quick environment check:

```powershell
@'
import torch
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
'@ | D:\miniconda\envs\myenv\python.exe -
```

## Required Data and Weights

Before running experiments, make sure these paths exist:

```text
data/PersonDataset
checkpoints/gasg_best.pth
checkpoints/depth_anything_v2_large.pth
third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth
third_party/ZeroStereo/checkpoint/hf_zerostereo
third_party/Depth-Anything-V2
third_party/MoGe
```

The best GASG checkpoint is protected. Current SHA256:

```text
251E2C67269A2D3054F554A8DD7B3C0F719405F2961FEFE46FB028B1E4B1D38E
```

## Reproduce the Paper Experiments

Run commands from the project root:

```powershell
cd D:\MyJupyter\Works\GASG_DEFOM_Stereo
```

### Step 1. Right-View Generation and GASG Ablations

This evaluates copy-left, constant shift, DA-V2 warp, MoGe warp, oracle warp, random GASG and GASG ablations.

```powershell
D:\miniconda\envs\myenv\python.exe scripts\eval_rightview_compare.py `
  --person_root data\PersonDataset `
  --save_dir results\rightview_generation `
  --height 256 --width 576 `
  --max_samples 200 `
  --device cuda `
  --gasg_ckpt checkpoints\gasg_best.pth `
  --gasg_gamma 1.0
```

### Step 2. Official ZeroStereo/StereoGen Right Views

If `data/zerostereo_person_test/` already contains generated right views, only evaluate them:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\baselines\eval_zerostereo_official.py `
  --work_root data\zerostereo_person_test `
  --save_dir results\rightview_generation `
  --height 256 --width 576 `
  --max_samples 200 `
  --evaluate_only
```

To regenerate ZeroStereo/StereoGen outputs with the official code, use:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\baselines\eval_zerostereo_official.py `
  --work_root data\zerostereo_person_test `
  --save_dir results\rightview_generation `
  --height 256 --width 576 `
  --max_samples 200 `
  --run_generation --force_prepare
```

### Step 3. Official StereoSpace HF Sample Check

StereoSpace is evaluated through its official Hugging Face Space. Public free GPU quota may stop the run early; the paper treats this as a small official hosted reproduction, not as the main 200-sample ranking.

```powershell
D:\miniconda\envs\myenv\python.exe scripts\baselines\eval_stereospace_hf.py `
  --work_root data\stereospace_person_test `
  --save_dir results\rightview_generation `
  --height 256 --width 576 `
  --max_samples 8 `
  --skip_existing --continue_on_error
```

### Step 4. Merge Right-View Results

```powershell
D:\miniconda\envs\myenv\python.exe scripts\merge_rightview_results.py `
  --results_dir results\rightview_generation
```

### Step 5. Pseudo-Stereo Depth Comparison

This fixes DEFOM-Stereo as the same downstream stereo estimator and changes only the right-view source.

```powershell
D:\miniconda\envs\myenv\python.exe scripts\run_person_depth_models.py `
  --methods defom_identity,defom_shift,defom_dav2_warp,defom_moge_warp,defom_zerostereo,gasg_defom,defom_real `
  --person_root data\PersonDataset `
  --save_dir results\pseudostereo_depth `
  --height 256 --width 576 `
  --max_samples 200 `
  --device cuda `
  --save_visuals `
  --skip_existing
```

StereoSpace depth uses only the saved official HF samples:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\eval_person_depth_model.py `
  --method defom_stereospace `
  --save_dir results\pseudostereo_depth `
  --height 256 --width 576 `
  --max_samples 2 `
  --save_visuals --num_visuals 2 `
  --stereospace_work_root data\stereospace_person_test `
  --use_saved_indices
```

Merge depth results:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\merge_person_depth_results.py `
  --results_dir results\pseudostereo_depth
```

### Step 6. Generate Paper Figures and Tables

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\miniconda\envs\myenv\python.exe scripts\figures\generate_paper_assets.py
```

Outputs:

```text
paper_assets/final_paper/figures/
paper_assets/final_paper/tables/
paper_assets/final_paper/data/
```

The same figures and tables are also copied into:

```text
paper_mdpi_applied_sciences/figures/
paper_mdpi_applied_sciences/tables/
```

### Step 7. Package the MDPI LaTeX Project

```powershell
Compress-Archive -Path paper_mdpi_applied_sciences\* `
  -DestinationPath GASG_MDPI_AppliedSciences_Paper.zip -Force
```

## Current Main Results

Right-view generation on PersonDataset:

| Method | N | PSNR | SSIM | LPIPS | Time |
|---|---:|---:|---:|---:|---:|
| ZeroStereo/StereoGen | 200 | 17.68 | 0.8163 | 0.1743 | 3492.7 ms |
| GASG | 200 | 28.56 | 0.9687 | 0.0425 | 155.6 ms |

Pseudo-stereo depth with DEFOM-Stereo as the common backend:

| Right-view source | N | AbsRel | RMSE | delta1 |
|---|---:|---:|---:|---:|
| ZeroStereo/StereoGen | 200 | 0.5354 | 7.2404 | 0.2605 |
| GASG | 200 | 0.2974 | 4.8595 | 0.3838 |
| True right reference | 200 | 0.2958 | 4.8371 | 0.3820 |

## Paper Assets

- Fig. 1: overall GASG pseudo-stereo pipeline
- Fig. 2: GASG architecture
- Fig. 3: right-view generation visual comparison
- Fig. 4: GASG ablation visual comparison
- Fig. 5: pseudo-stereo depth prediction comparison
- Fig. 6: absolute depth error comparison
- Fig. 7: right-view PSNR/SSIM/LPIPS bars
- Fig. 8: pseudo-stereo depth AbsRel/RMSE/delta1 bars
- Table 1: right-view generation quality
- Table 2: pseudo-stereo depth comparison
- Table 3: GASG ablation

## Optional: Train GASG Again

The provided checkpoint is the one used for the paper. If you want to train again without overwriting it:

```powershell
D:\miniconda\envs\myenv\python.exe scripts\train_gasg.py `
  --config configs\gasg_config.yaml `
  --person_root data\PersonDataset `
  --exp_suffix reproduce `
  --num_workers 4
```

Training outputs are written under `experiments/` if you run this command. Keep the paper checkpoint at `checkpoints/gasg_best.pth` unchanged unless you intentionally replace it.
