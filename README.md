# GASG: Pseudo-Stereo Depth from a Single Camera

GASG is a lightweight left-to-right image generator trained for pseudo-stereo depth estimation. At inference time, a single physical camera provides only the left image. GASG synthesizes a pseudo right image, and the pair can be passed to any stereo depth estimator that accepts rectified left-right inputs.

    left image -> GASG -> pseudo right image
    left image + pseudo right image -> stereo depth estimator -> absolute depth

The contribution is the GASG generator and its plug-in pseudo-stereo interface. `GASG -> DEFOM-Stereo` is only one representative evaluation pipeline, not the proposed method name.

## Project Layout

    checkpoints/
      gasg_best.pth                         # best GASG checkpoint, do not delete
      depth_anything_v2_large.pth           # DA-V2 dependency
    
    configs/
      gasg_config.yaml
      eval_config.yaml
    
    data/
      PersonDataset/                        # train/test left, right and metric depth
      zerostereo_person_tuned_test/         # PersonDataset-fine-tuned ZeroStereo outputs
      hf_cache/                             # Hugging Face cache for MoGe and related models
    
    models/
      gasg_net.py                           # GASG model
    
    scripts/
      train_gasg.py
      calibrate_warp_baselines.py           # calibrate DA-V2/MoGe warp scales on train split
      eval_rightview_compare.py             # right-view metrics, speed and GASG ablations
      merge_rightview_results.py
      eval_person_depth_model.py
      run_person_depth_models.py
      merge_person_depth_results.py
      baselines/eval_zerostereo_official.py # official ZeroStereo/StereoGen reproduction
      figures/create_overall_pipeline_fig.py
      figures/generate_paper_assets.py
    
    third_party/
      DEFOM-Stereo/
      Depth-Anything-V2/
      MoGe/
      ZeroStereo/
    
    results/
      rightview_generation/                 # final right-view experiment results
      pseudostereo_depth/                   # final pseudo-stereo depth results
    
    paper_assets/final_paper/               # selected figures, tables and CSV summaries
    paper_mdpi_applied_sciences/            # MDPI LaTeX manuscript package

## Environment

The experiments in this workspace use:

    D:\miniconda\envs\myenv\python.exe

Quick check:

    @'
    import torch
    print(torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    '@ | D:\miniconda\envs\myenv\python.exe -

## Required Files

Make sure the following paths exist before reproducing all experiments:

    data/PersonDataset
    checkpoints/gasg_best.pth
    checkpoints/depth_anything_v2_large.pth
    third_party/DEFOM-Stereo/checkpoints/defomstereo_vitl_sceneflow.pth
    third_party/ZeroStereo/checkpoint/hf_zerostereo
    third_party/ZeroStereo/checkpoint/stereogen_person/final
    third_party/Depth-Anything-V2
    third_party/MoGe

The protected GASG checkpoint used in the paper is:

    checkpoints/gasg_best.pth
    SHA256: 251E2C67269A2D3054F554A8DD7B3C0F719405F2961FEFE46FB028B1E4B1D38E

Public links:

* Code: https://github.com/jibingyang11/GASG/tree/main
* PersonDataset: https://huggingface.co/datasets/jibingyang111/PersonDataset

## Reproduce the Paper Experiments

Run all commands from the project root:

    cd D:\MyJupyter\Works\GASG_DEFOM_Stereo

### 1. Calibrate Monocular-Warp Baselines

DA-V2 and MoGe are not used as raw zero-shot warps. Their scalar disparity ranges are calibrated on the PersonDataset training split and then fixed for test evaluation.

    D:\miniconda\envs\myenv\python.exe scripts\calibrate_warp_baselines.py `
      --person_root data\PersonDataset `
      --save_dir results\rightview_generation `
      --height 256 --width 576 `
      --max_samples 96 `
      --device cuda `
      --dav2_grid 84:144:12 `
      --moge_grid 42:84:6

Current calibrated values:

    DA-V2 target_disp = 96
    MoGe target_disp  = 78

### 2. Fine-Tune ZeroStereo/StereoGen on PersonDataset

This uses the official ZeroStereo/StereoGen training code with the PersonDataset adapter in `third_party/ZeroStereo/dataset/inpaint_dataset.py`.

    Set-Location third_party\ZeroStereo
    D:\miniconda\envs\myenv\python.exe train_stereogen.py `
      --config config\train_stereogen_person.yaml `
      --trainer.total_steps 120 `
      --trainer.save_path checkpoint\stereogen_person
    Set-Location ..\..

The resulting fine-tuned pipeline is expected at:

    third_party/ZeroStereo/checkpoint/stereogen_person/final

### 3. Evaluate Right-View Generation and GASG Ablations

    D:\miniconda\envs\myenv\python.exe scripts\eval_rightview_compare.py `
      --person_root data\PersonDataset `
      --save_dir results\rightview_generation `
      --height 256 --width 576 `
      --max_samples 200 `
      --device cuda `
      --gasg_ckpt checkpoints\gasg_best.pth `
      --gasg_gamma 1.0 `
      --dav2_target_disp 96 `
      --moge_target_disp 78 `
      --methods identity,shift_const,dav2_warp,moge_warp,oracle_warp,gasg_untrained,gasg_warp_only,gasg_direct_only,gasg_no_refine,gasg_full

### 4. Generate and Evaluate Fine-Tuned ZeroStereo/StereoGen Right Views

    D:\miniconda\envs\myenv\python.exe scripts\baselines\eval_zerostereo_official.py `
      --person_root data\PersonDataset `
      --work_root data\zerostereo_person_tuned_test `
      --filelist_name person_tuned_eval_200.txt `
      --save_dir results\rightview_generation `
      --height 256 --width 576 `
      --max_samples 200 `
      --num_inference_step 20 `
      --method_name zerostereo_person_tuned `
      --stereogen_model_path third_party\ZeroStereo\checkpoint\stereogen_person\final `
      --run_generation `
      --force_prepare

### 5. Merge Right-View Metrics

    D:\miniconda\envs\myenv\python.exe scripts\merge_rightview_results.py `
      --results_dir results\rightview_generation

### 6. Evaluate Pseudo-Stereo Depth

All rows use DEFOM-Stereo as the same downstream stereo estimator. Only the source of the right view changes.

    D:\miniconda\envs\myenv\python.exe scripts\run_person_depth_models.py `
      --methods defom_identity,defom_shift,defom_dav2_warp,defom_moge_warp,defom_zerostereo_tuned,gasg_defom,defom_real `
      --person_root data\PersonDataset `
      --save_dir results\pseudostereo_depth `
      --height 256 --width 576 `
      --max_samples 200 `
      --device cuda `
      --save_visuals `
      --dav2_target_disp 96 `
      --moge_target_disp 78 `
      --zerostereo_tuned_work_root data\zerostereo_person_tuned_test

### 7. Generate Paper Figures and Tables

    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    D:\miniconda\envs\myenv\python.exe scripts\figures\generate_paper_assets.py

Outputs:

    paper_assets/final_paper/figures/
    paper_assets/final_paper/tables/
    paper_assets/final_paper/data/
    paper_mdpi_applied_sciences/figures/
    paper_mdpi_applied_sciences/tables/

### 8. Package the MDPI LaTeX Project

    Compress-Archive -Path paper_mdpi_applied_sciences\* `
      -DestinationPath GASG_MDPI_AppliedSciences_Paper.zip -Force

## Current Fair-Protocol Results

Right-view generation on 200 PersonDataset test samples:

| Method | PSNR | SSIM | LPIPS | Time |
| --- | --- | --- | --- | --- |
| Copy-left | 16.15 | 0.7753 | 0.2858 | 0.3 ms |
| Constant shift | 16.82 | 0.8091 | 0.2222 | 1.3 ms |
| DA-V2 calibrated warp | 18.07 | 0.8300 | 0.1916 | 467.2 ms |
| MoGe calibrated warp | 18.26 | 0.8346 | 0.1748 | 305.2 ms |
| ZeroStereo/StereoGen fine-tuned | 17.56 | 0.8124 | 0.1801 | 1765.7 ms |
| GASG | 28.56 | 0.9687 | 0.0425 | 167.0 ms |

Pseudo-stereo depth with DEFOM-Stereo as the common backend:

| Right-view source | AbsRel | RMSE | delta1 |
| --- | --- | --- | --- |
| Copy-left | 1.0727 | 12.0946 | 0.0570 |
| Constant shift | 1.3460 | 14.0147 | 0.0178 |
| DA-V2 calibrated warp | 0.3830 | 5.6722 | 0.2458 |
| MoGe calibrated warp | 0.3211 | 5.0402 | 0.3450 |
| ZeroStereo/StereoGen fine-tuned | 0.5356 | 7.2431 | 0.2604 |
| GASG | 0.2974 | 4.8595 | 0.3838 |
| True right reference | 0.2958 | 4.8371 | 0.3820 |

## Paper Assets

* Fig. 1: overall GASG pseudo-stereo pipeline
* Fig. 2: GASG architecture
* Fig. 3: right-view generation visual comparison
* Fig. 4: GASG ablation visual comparison
* Fig. 5: pseudo-stereo depth prediction comparison
* Fig. 6: absolute depth error comparison
* Fig. 7: right-view PSNR/SSIM/LPIPS bars
* Fig. 8: pseudo-stereo depth AbsRel/RMSE/delta1 bars
* Table 1: right-view generation metrics
* Table 2: pseudo-stereo depth metrics
* Table 3: GASG ablation metrics

## Notes

StereoSpace and StereoDiffusion are cited and discussed as recent pseudo-stereo generation methods, but they are not included in the main fair-protocol table because the available official releases in this workspace did not provide a comparable local fine-tuning path. The main quantitative comparison therefore uses reproducible baselines that can be adapted or calibrated on PersonDataset.
