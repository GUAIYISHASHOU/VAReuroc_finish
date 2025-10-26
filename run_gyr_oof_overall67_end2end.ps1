$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Route = "gyr"
$CacheRoot = "F:\SLAMdata\_cache\imu"
$TrainNPZ = Join-Path $CacheRoot "train_gyr.npz"
$TestNPZ  = Join-Path $CacheRoot "test_gyr_mh03.npz"
$SaveRoot = "runs\gyr_oof_ep200"
$K = 4
$KJson = Join-Path $SaveRoot ("k{0}.json" -f $K)
$CalibOut = Join-Path $SaveRoot "calibrator_oof_offset_overall67_gauss.json"
$PlotsDir = Join-Path $SaveRoot "mh03_plots_overall67_gauss_foldsc"

New-Item -ItemType Directory -Force -Path $SaveRoot | Out-Null

python -m tools.make_kfold_splits `
  --npz $TrainNPZ `
  --out_json $KJson `
  --k $K

if (-not $env:SKIP_TRAIN) {
  python -m tools.train_oof `
    --route $Route `
    --npz $TrainNPZ `
    --kfold_json $KJson `
    --save_root $SaveRoot `
    --epochs 200 `
    --batch 64 `
    --d_model 128 `
    --n_tcn 4 `
    --kernel 5 `
    --nu 6.0 `
    --logv_min -10 `
    --logv_max 4 `
    --early_stop_patience 15 `
    --early_stop_delta 0.0 `
    --scheduler plateau
}

python -m tools.make_oof_offset_calib `
  --route $Route `
  --npz $TrainNPZ `
  --kfold_json $KJson `
  --save_root $SaveRoot `
  --out_json $CalibOut `
  --target_cov_overall `
  --target_cov 0.67

python -m tools.eval_imu `
  --route $Route `
  --npz $TestNPZ `
  --ckpt_glob (Join-Path $SaveRoot "fold*\best.pt") `
  --calib_json $CalibOut `
  --plots_dir $PlotsDir `
  --use_fold_scalers `
  --also_overall `
  --compare_uncalibrated `
  --apply_mask
