# IMU Uncertainty (VIS-style)

End-to-end training, calibration, and evaluation pipelines for VIS-style IMU uncertainty models.

See scripts under `tools/` and package `imu/`.

## Prerequisites

- Activate your conda environment (e.g., LAP3GPU).
- Ensure the IMU cache exists at `F:\SLAMdata\_cache\imu` with:
  - `train_acc.npz`, `test_acc_mh_03.npz`
  - `train_gyr.npz`, `test_gyr_mh03.npz`
- Windows PowerShell is recommended.

## One-click pipelines

Run from the repo root or the script’s folder:

```powershell
# Accelerometer: 4-fold OOF training -> offset calibration (overall 0.68) -> evaluation on MH_03
.\run_acc_oof_overall68_end2end.ps1

# Gyroscope: 4-fold OOF training -> offset calibration (overall 0.67) -> evaluation on MH_03
.\run_gyr_oof_overall67_end2end.ps1
```

To reuse existing checkpoints and only run calibration/eval:

```powershell
$env:SKIP_TRAIN = 1
.\run_acc_oof_overall68_end2end.ps1
.\run_gyr_oof_overall67_end2end.ps1
$env:SKIP_TRAIN = $null
```

## What the scripts do

- Make K-fold splits:
  - `python -m tools.make_kfold_splits --npz <train.npz> --out_json <k.json> --k 4`
- Train OOF models:
  - `python -m tools.train_oof --route <acc|gyr> --npz <train.npz> --kfold_json <k.json> --save_root <runs/...> --epochs 200 ...`
- Fit offset calibrator targeting overall coverage:
  - Accelerometer: `--target_cov 0.68`
  - Gyroscope: `--target_cov 0.67`
- Evaluate:
  - `python -m tools.eval_imu --route <acc|gyr> --npz <test.npz> --ckpt_glob <runs/.../fold*/best.pt> --calib_json <...json> --plots_dir <...> --use_fold_scalers --also_overall --compare_uncalibrated --apply_mask`

## Outputs

- Accelerometer (`runs\acc_oof_ep200`):
  - `calibrator_oof_offset_overall68.json`
  - `mh_03_plots_overall68_foldsc\` and per-fold checkpoints in `fold*\best.pt`
- Gyroscope (`runs\gyr_oof_ep200`):
  - `calibrator_oof_offset_overall67_gauss.json`
  - `mh03_plots_overall67_gauss_foldsc\` and per-fold checkpoints in `fold*\best.pt`

## Adjusting paths/hyperparameters

Edit the corresponding `.ps1` to change:

- Cache root and dataset filenames
- `K`, `epochs`, `batch`, `d_model`, `n_tcn`, `kernel`, `nu`, `logv_min/max`
- Target coverage and output directories
