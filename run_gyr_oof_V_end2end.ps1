$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Route = "gyr"
$EachSeqDir = "F:\SLAMdata\_cache\imu_eachseq_V"
$SplitDir   = "F:\SLAMdata\_cache\imu_split_V"
$SaveRoot = "runs\gyr_oof_V_b104_ep200"
$K = 4
$KJson = Join-Path $SaveRoot ("k{0}.json" -f $K)
$CalibOutV1 = Join-Path $SaveRoot "calibrator_oof_offset_overall69.json"
$CalibOutV2 = Join-Path $SaveRoot "calibrator_oof_offset_overall67.json"
$UseDyn = $true
$DynAxes = $true
$FeatDt = 0.005
$FeatWin = 128
$FeatHfFc = 0.3
$FeatFFT = 128

# Build dynamic feature args as an array so PowerShell expands into separate argv tokens
$DynArgs = @()
if ($UseDyn) {
  $DynArgs = @("--use_dyn_feats","--feat_dt",$FeatDt,"--feat_win",$FeatWin,"--feat_hf_fc",$FeatHfFc,"--feat_fft_n",$FeatFFT)
  if ($DynAxes) { $DynArgs += "--dyn_axes" }
}

New-Item -ItemType Directory -Force -Path $SaveRoot | Out-Null
New-Item -ItemType Directory -Force -Path $SplitDir | Out-Null

# Merge from a single directory; assign whole sequences to test via exclusion
python -m tools.split_eachseq_merge_npz `
  --root $EachSeqDir `
  --out_dir $SplitDir `
  --pattern "*.npz" `
  --route imu `
  --split 1.0 0.0 0.0 `
  --exclude_seqs "V1_03_difficult,V2_02_medium"

$TrainNPZ = (Join-Path $SplitDir "train_all.npz")

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
    --nu 4.0 `
    --logv_min -10 `
    --logv_max 5 `
    --early_stop_patience 15 `
    --early_stop_delta 0.0 `
    --scheduler plateau `
    $DynArgs
}

# Generate two overall beta calibrators with different targets for each test seq
python -m tools.make_oof_offset_calib `
  --route $Route `
  --npz $TrainNPZ `
  --kfold_json $KJson `
  --save_root $SaveRoot `
  --out_json $CalibOutV1 `
  --target_cov_overall `
  --target_cov 0.69 `
  --apply_mask `
  $DynArgs

python -m tools.make_oof_offset_calib `
  --route $Route `
  --npz $TrainNPZ `
  --kfold_json $KJson `
  --save_root $SaveRoot `
  --out_json $CalibOutV2 `
  --target_cov_overall `
  --target_cov 0.67 `
  --apply_mask `
  $DynArgs

# Per-sequence evaluation on the two test sequences
$TestSeq1 = "F:\SLAMdata\_cache\imu_eachseq_V\V1_03_difficult_T512_S256.npz"
$TestSeq2 = "F:\SLAMdata\_cache\imu_eachseq_V\V2_02_medium_T512_S256.npz"
$Plots1 = Join-Path $SaveRoot "V1_03_difficult_plots_overall69_foldsc"
$Plots2 = Join-Path $SaveRoot "V2_02_medium_plots_overall67_foldsc"

python -m tools.eval_imu `
  --route $Route `
  --npz $TestSeq1 `
  --ckpt_glob (Join-Path $SaveRoot "fold*\best.pt") `
  --calib_json $CalibOutV1 `
  --plots_dir $Plots1 `
  --use_fold_scalers `
  --also_overall `
  --compare_uncalibrated `
  --apply_mask `
  --bound_kind none `
  --dyn_dual `
  $DynArgs

python -m tools.eval_imu `
  --route $Route `
  --npz $TestSeq2 `
  --ckpt_glob (Join-Path $SaveRoot "fold*\best.pt") `
  --calib_json $CalibOutV2 `
  --plots_dir $Plots2 `
  --use_fold_scalers `
  --also_overall `
  --compare_uncalibrated `
  --apply_mask `
  --bound_kind none `
  --dyn_dual `
  $DynArgs
