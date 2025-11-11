$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Route = "gyr"
$EachSeqRoot = "F:\SLAMdata\_cache\imu_eachseq_MH"
$Pattern = "MH_*_T*_S*.npz"
$SaveBase = "runs\gyr_oof_mh01_mh05_ep200"
$K = 4

$ValSeqs = @("MH_01_easy", "MH_04_difficult", "MH_05_difficult")

foreach ($Val in $ValSeqs) {
  $ExpRoot  = Join-Path $SaveBase $Val
  $SplitDir = Join-Path $ExpRoot "split"
  $SaveRoot = Join-Path $ExpRoot "oof"
  New-Item -ItemType Directory -Force -Path $SaveRoot | Out-Null

  python -m tools.split_eachseq_merge_npz `
    --root $EachSeqRoot `
    --out_dir $SplitDir `
    --pattern $Pattern `
    --route imu `
    --test_seqs $Val

  $TrainNPZ = Join-Path $SplitDir "train_all.npz"
  $TestNPZ  = Join-Path $SplitDir "test_all.npz"
  $KJson    = Join-Path $SaveRoot ("k{0}.json" -f $K)
  $CalibOut = Join-Path $SaveRoot "calibrator_oof_offset_overall67_mh.json"
  $PlotsDir = Join-Path $SaveRoot ("plots_" + $Val + "_foldsc")

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
    --use_student_t_thr `
    --nu 6.0 `
    --apply_mask
}
