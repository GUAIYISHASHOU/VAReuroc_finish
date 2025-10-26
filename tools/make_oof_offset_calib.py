import argparse, json, os, numpy as np, torch
from torch.utils.data import DataLoader
from imu.datasets_imu_frames import IMUFrames
from imu.model_sa3 import IMUSA3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--kfold_json", required=True)
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--target_cov", type=float, default=0.68)
    ap.add_argument("--cov_thresh", type=float, default=1.046**2)
    ap.add_argument("--lambda_scale", type=float, default=1.0)
    ap.add_argument("--beta_min", type=float, default=0.0)
    ap.add_argument("--target_cov_overall", action="store_true")
    args = ap.parse_args()

    import json as _json
    KF = _json.load(open(args.kfold_json, "r", encoding="utf-8"))["folds"]

    base = IMUFrames(args.npz, route=args.route, scaler_npz=None)
    base_seq = base.seq

    all_pred = []
    all_gt = []

    for i, fold in enumerate(KF):
        scaler = os.path.join(args.save_root, f"fold{i}", "scaler.npz")
        ckpt_p = os.path.join(args.save_root, f"fold{i}", "best.pt")
        va_idx = np.where(np.isin(base_seq, fold["val"]))[0]
        ds = IMUFrames(args.npz, route=args.route, scaler_npz=scaler, subset_ids=va_idx)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        ck = torch.load(ckpt_p, map_location="cpu")
        mdl = IMUSA3(d_in=ck["d_in"], d_model=ck["args"]["d_model"], n_tcn=ck["args"]["n_tcn"], k=ck["args"]["kernel"], n_tf=ck["args"]["n_tf"], logv_min=ck["args"]["logv_min"], logv_max=ck["args"]["logv_max"]) 
        mdl.load_state_dict(ck["model"]) 
        mdl.eval()
        with torch.no_grad():
            preds=[]; gts=[]
            for b in dl:
                logv,_ = mdl(b["X"]) 
                preds.append(logv.numpy())
                gts.append((b["E"].numpy()**2))
        all_pred.append(np.concatenate(preds,0))
        all_gt.append(np.concatenate(gts,0))

    P = np.concatenate(all_pred,0)
    T = np.concatenate(all_gt,0)
    z2 = T / (np.exp(P) + 1e-12)
    # Only keep overall-coverage calibration (overall@target_cov)
    tgt = float(args.target_cov)
    thr = float(args.cov_thresh)
    bmin = float(args.beta_min)
    var = np.exp(P)
    var_o = np.mean(var, axis=-1, keepdims=True)
    P_o = np.log(var_o + 1e-12)  # (N,T,1)
    T_o = np.mean(T, axis=-1, keepdims=True)
    def cov_of(beta: float) -> float:
        z2o = T_o / (np.exp(P_o + beta) + 1e-12)
        return float((z2o <= thr).mean())
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = 0.5*(lo+hi)
        if cov_of(mid) >= tgt:
            hi = mid
        else:
            lo = mid
    beta = 0.5*(lo+hi)
    if np.isfinite(bmin):
        beta = max(beta, bmin)
    betas = np.array([beta, beta, beta], dtype=np.float64)
    mode_used = f"cov_overall@{tgt}"

    J = {"route": args.route,
         "axes": [{"alpha": 1.0, "beta": float(b)} for b in betas.tolist()],
         "info": {"source": "oof_offset", "mode": mode_used, "lambda": 1.0, "beta_min": bmin}}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(J, f, indent=2, ensure_ascii=False)
    print("Saved", args.out_json)

if __name__ == "__main__":
    main()
