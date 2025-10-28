# OOF (Out-of-Fold) Calibration Workflow for V-series

$ErrorActionPreference = "Stop"

# ============================================================
# Configuration
# ============================================================
$TRAIN_NPZ   = "F:/SLAMdata/_cache/Vvis/train_v.npz"
$GEOM_STATS  = "F:/SLAMdata/_cache/Vvis/geom_stats_24d.npz"
$TEST1_NPZ   = "F:/SLAMdata/_cache/Vvis/V1_03_difficult.npz"   # test seq 1
$TEST2_NPZ   = "F:/SLAMdata/_cache/Vvis/V2_02_medium.npz"      # test seq 2
$SAVE_ROOT   = "runs/oof_Vseries"
$K_FOLDS     = 5

# ============================================================
# Step 1: Generate K-fold splits (run once)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 1: Generate K-fold splits" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$KFOLD_JSON = "$SAVE_ROOT/k${K_FOLDS}_splits.json"

if (Test-Path $KFOLD_JSON) {
    Write-Host "K-fold split file exists: $KFOLD_JSON" -ForegroundColor Yellow
    Write-Host "Skipping generation..." -ForegroundColor Yellow
} else {
    python tools/make_kfold_splits.py --train_npz $TRAIN_NPZ --out_json $KFOLD_JSON --k $K_FOLDS --seed 42
    if ($LASTEXITCODE -ne 0) { Write-Host "Error: K-fold split failed" -ForegroundColor Red; exit 1 }
}

# ============================================================
# Step 2: OOF training and calibrator fitting (train only)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: OOF training and calibrator fitting" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Note: This will train $K_FOLDS models, may take a while..." -ForegroundColor Yellow

python tools/train_oof.py `
  --train_npz $TRAIN_NPZ `
  --geom_stats_npz $GEOM_STATS `
  --kfold_json $KFOLD_JSON `
  --save_root $SAVE_ROOT `
  --epochs 40 --stage1_epochs 8 `
  --batch_size 32 --lr 2e-4 `
  --a_max 3.0 --drop_token_p 0.1 `
  --heads 4 --layers 1 --d_model 128 `
  --nll_weight 1.5 --bce_weight 0.6 --rank_weight 0.3 `
  --patience 12 `
  --sa_mode deming --deming_lambda 1.0

if ($LASTEXITCODE -ne 0) { Write-Host "Error: OOF training failed" -ForegroundColor Red; exit 1 }

$CALIBRATOR_JSON = "$SAVE_ROOT/calibrator_oof.json"
$BEST_CKPT = "$SAVE_ROOT/fold0/best_macro_sa.pt"
Write-Host "`nOOF calibrator generated: $CALIBRATOR_JSON" -ForegroundColor Green

# ============================================================
# Step 3: Evaluate on Test sequence 1 (apply OOF calibration)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Evaluate on V1_03_difficult (Test1)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$PLOT_DIR_1 = "$SAVE_ROOT/test_V1_03"
python eval_macro.py `
  --npz $TEST1_NPZ `
  --ckpt $BEST_CKPT `
  --geom_stats_npz $GEOM_STATS `
  --calibrator_json $CALIBRATOR_JSON `
  --kappa 1.0 --sa_recenter --scan_q_threshold `
  --plots_dir $PLOT_DIR_1

if ($LASTEXITCODE -ne 0) { Write-Host "Error: Test1 evaluation failed" -ForegroundColor Red; exit 1 }

# ============================================================
# Step 4: Evaluate on Test sequence 2 (apply OOF calibration)
# ============================================================
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 4: Evaluate on V2_02_medium (Test2)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$PLOT_DIR_2 = "$SAVE_ROOT/test_V2_02"
python eval_macro.py `
  --npz $TEST2_NPZ `
  --ckpt $BEST_CKPT `
  --geom_stats_npz $GEOM_STATS `
  --calibrator_json $CALIBRATOR_JSON `
  --kappa 1.0 --sa_recenter --scan_q_threshold `
  --plots_dir $PLOT_DIR_2

if ($LASTEXITCODE -ne 0) { Write-Host "Error: Test2 evaluation failed" -ForegroundColor Red; exit 1 }

# ============================================================
# Done
# ============================================================
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "OOF V-series workflow completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nResults Root: $SAVE_ROOT" -ForegroundColor Yellow
Write-Host "  - OOF calibrator: $CALIBRATOR_JSON" -ForegroundColor White
Write-Host "  - Test1 results: $PLOT_DIR_1" -ForegroundColor White
Write-Host "  - Test2 results: $PLOT_DIR_2" -ForegroundColor White
