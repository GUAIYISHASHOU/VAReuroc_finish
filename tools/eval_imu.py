
import argparse, json, os, glob, numpy as np, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from imu.datasets_imu_frames import IMUFrames
from imu.model_sa3 import IMUSA3

def spearman_np(x,y):
    rx = x.argsort().argsort().astype(np.float64); ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = (rx**2).sum()**0.5 * (ry**2).sum()**0.5 + 1e-12
    return float((rx*ry).sum() / den)

ap = argparse.ArgumentParser()
ap.add_argument("--route", choices=["acc","gyr"], required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--ckpts", nargs="+", default=None)
ap.add_argument("--ckpt_glob", default=None)
ap.add_argument("--geom_stats", default=None)
ap.add_argument("--calib_json", default=None)
ap.add_argument("--plots_dir", default=None)
ap.add_argument("--also_overall", action="store_true")
ap.add_argument("--save_pred_npz", default=None)
ap.add_argument("--compare_uncalibrated", action="store_true")
ap.add_argument("--plot_inliers_only", action="store_true")
ap.add_argument("--inlier_hi", type=float, default=3.0**2)
ap.add_argument("--inlier_lo", type=float, default=1.0/(3.0**2))
ap.add_argument("--inlier_thr", type=float, default=3.0**2)  # backward compat (hi only)
ap.add_argument("--shape_clip", choices=["none","ellipse","band"], default="none")
ap.add_argument("--shape_k", type=float, default=2.5)
ap.add_argument("--shape_bins", type=int, default=48)
ap.add_argument("--shape_uq", type=float, default=0.99)
ap.add_argument("--shape_soft", type=float, default=0.15)  # 软过渡占比(0~1)
ap.add_argument("--shape_remove", action="store_true")
ap.add_argument("--resid_remove", action="store_true")
ap.add_argument("--resid_q", type=float, default=0.25)
ap.add_argument("--resid_k", type=float, default=2.5)
ap.add_argument("--scaler_npz", default=None)
ap.add_argument("--use_fold_scalers", action="store_true")
ap.add_argument("--use_student_t_thr", action="store_true")
ap.add_argument("--nu", type=float, default=None)
ap.add_argument("--apply_mask", action="store_true")
ap.add_argument("--logv_min_override", type=float, default=None)
ap.add_argument("--logv_max_override", type=float, default=None)
ap.add_argument("--drop_outliers", action="store_true")
ap.add_argument("--rm_box_var_lo", type=float, default=None)
ap.add_argument("--rm_box_var_hi", type=float, default=None)
ap.add_argument("--rm_box_e2_lo", type=float, default=None)
ap.add_argument("--rm_box_e2_hi", type=float, default=None)
ap.add_argument("--rm_ell_var_c", type=float, default=None)
ap.add_argument("--rm_ell_e2_c", type=float, default=None)
ap.add_argument("--rm_ell_ru", type=float, default=None)
ap.add_argument("--rm_ell_rv", type=float, default=None)
args = ap.parse_args()

# Determine checkpoints list (support ensemble)
ckpt_list = []
if args.ckpt_glob:
    ckpt_list.extend(sorted(glob.glob(args.ckpt_glob)))
if args.ckpts:
    ckpt_list.extend(list(args.ckpts))
if (not ckpt_list) and args.ckpt:
    ckpt_list = [args.ckpt]
if not ckpt_list:
    raise ValueError("Please specify --ckpt or --ckpts or --ckpt_glob")

# Coverage thresholds (default: Gaussian-like refs)
thr68 = float(1.046**2)
thr95 = float(3.0**2)

def _student_t_thresholds(nu_val: float):
    try:
        df = torch.tensor(float(nu_val))
        dist = torch.distributions.StudentT(df=df)
        q68 = torch.abs(dist.icdf(torch.tensor(0.84))).item()
        q95 = torch.abs(dist.icdf(torch.tensor(0.975))).item()
        return float(q68**2), float(q95**2)
    except Exception:
        # Fallback via sampling approximation
        df = torch.tensor(float(nu_val))
        dist = torch.distributions.StudentT(df=df)
        smp = dist.sample((200000,))
        q68 = torch.quantile(torch.abs(smp), 0.84).item()
        q95 = torch.quantile(torch.abs(smp), 0.975).item()
        return float(q68**2), float(q95**2)

if len(ckpt_list) == 1:
    ck = torch.load(ckpt_list[0], map_location="cpu")
    scaler_single = args.scaler_npz
    if not scaler_single:
        cand = os.path.join(os.path.dirname(ckpt_list[0]), "scaler.npz")
        if os.path.exists(cand):
            scaler_single = cand
    ds = IMUFrames(args.npz, route=args.route, scaler_npz=scaler_single)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    if args.use_student_t_thr:
        nu_val = float(args.nu) if args.nu is not None else float(ck.get("args", {}).get("nu", 6.0))
        thr68, thr95 = _student_t_thresholds(nu_val)
    lvmin = ck["args"].get("logv_min", -10.0)
    lvmax = ck["args"].get("logv_max", 4.0)
    if args.logv_min_override is not None:
        lvmin = float(args.logv_min_override)
    if args.logv_max_override is not None:
        lvmax = float(args.logv_max_override)
    mdl = IMUSA3(d_in=ck["d_in"], d_model=ck["args"]["d_model"], n_tcn=ck["args"]["n_tcn"],
                 k=ck["args"]["kernel"], n_tf=ck["args"]["n_tf"],
                 logv_min=lvmin, logv_max=lvmax)
    mdl.load_state_dict(ck["model"]); mdl.eval()
    preds=[]; E2s=[]; Ms=[]
    with torch.no_grad():
        for b in dl:
            logv,_ = mdl(b["X"])
            preds.append(logv.numpy())
            E2s.append((b["E"].numpy()**2))
            Ms.append(b["M"].numpy())
    P = np.concatenate(preds,0)
    T = np.concatenate(E2s,0)
    M = np.concatenate(Ms,0)
else:
    preds_all = []
    T = None
    Ms = []
    for idx, ckpt_p in enumerate(ckpt_list):
        ck = torch.load(ckpt_p, map_location="cpu")
        lvmin = ck["args"].get("logv_min", -10.0)
        lvmax = ck["args"].get("logv_max", 4.0)
        if args.logv_min_override is not None:
            lvmin = float(args.logv_min_override)
        if args.logv_max_override is not None:
            lvmax = float(args.logv_max_override)
        mdl = IMUSA3(d_in=ck["d_in"], d_model=ck["args"]["d_model"], n_tcn=ck["args"]["n_tcn"],
                     k=ck["args"]["kernel"], n_tf=ck["args"]["n_tf"],
                     logv_min=lvmin, logv_max=lvmax)
        mdl.load_state_dict(ck["model"]); mdl.eval()
        if idx == 0 and args.use_student_t_thr:
            nu_val = float(args.nu) if args.nu is not None else float(ck.get("args", {}).get("nu", 6.0))
            thr68, thr95 = _student_t_thresholds(nu_val)
        scaler_p = None
        if args.use_fold_scalers:
            cand = os.path.join(os.path.dirname(ckpt_p), "scaler.npz")
            if os.path.exists(cand):
                scaler_p = cand
        ds = IMUFrames(args.npz, route=args.route, scaler_npz=scaler_p)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        preds=[]; E2s=[]
        with torch.no_grad():
            for b in dl:
                logv,_ = mdl(b["X"])
                preds.append(logv.numpy())
                if idx == 0:
                    E2s.append((b["E"].numpy()**2))
                    Ms.append(b["M"].numpy())
        preds_all.append(np.concatenate(preds,0))
        if idx == 0:
            T = np.concatenate(E2s,0)
    var_ens = np.mean([np.exp(Pi) for Pi in preds_all], axis=0)
    P = np.log(var_ens + 1e-12)
    M = np.concatenate(Ms,0)
logE2 = np.log(T + 1e-12)
P_raw = P.copy()

if args.calib_json and os.path.exists(args.calib_json):
    J = json.load(open(args.calib_json,"r",encoding="utf-8"))
    for j in range(3):
        a = J["axes"][j]["alpha"]; b = J["axes"][j]["beta"]
        P[...,j] = a * P[...,j] + b

base_mask = (M > 0.5) if args.apply_mask else np.ones_like(M, dtype=bool)
# optional drop of outliers based on overall z2 window
if args.drop_outliers:
    var_o_m = np.mean(np.exp(P), axis=-1, keepdims=True)
    P_o_m = np.log(var_o_m + 1e-12)
    T_o_m = np.mean(T, axis=-1, keepdims=True)
    z2_o_m = (T_o_m / (np.exp(P_o_m) + 1e-12))
    lo_m = float(getattr(args, 'inlier_lo', 1.0/(3.0**2)))
    hi_m = float(getattr(args, 'inlier_hi', 3.0**2))
    keep = (z2_o_m >= lo_m) & (z2_o_m <= hi_m)
    base_mask = base_mask & keep.squeeze(-1)
mask = base_mask if (args.apply_mask or args.drop_outliers) else None
if mask is not None:
    spears = [spearman_np(P[...,j][mask].ravel(), logE2[...,j][mask].ravel()) for j in range(3)]
else:
    spears = [spearman_np(P[...,j].ravel(), logE2[...,j].ravel()) for j in range(3)]
z2 = (T / (np.exp(P) + 1e-12))
if mask is not None:
    m3 = np.repeat(mask[..., None], 3, axis=-1)
    z2_mean = float(np.mean(z2[m3]))
    cov68 = float(np.mean((z2 <= thr68)[m3]))
    cov95 = float(np.mean((z2 <= thr95)[m3]))
else:
    z2_mean = float(np.mean(z2))
    cov68 = float(np.mean((z2 <= thr68)))
    cov95 = float(np.mean((z2 <= thr95)))

metrics = {
    "spearman_axes": spears,
    "spearman_mean": float(np.mean(spears)),
    "z2_mean": z2_mean,
    "cov68": cov68,
    "cov95": cov95,
}

if args.also_overall:
    var = np.exp(P)
    var_o = np.mean(var, axis=-1, keepdims=True)           # (N,T,1)
    P_o = np.log(var_o + 1e-12)
    T_o = np.mean(T, axis=-1, keepdims=True)
    logE2_o = np.log(T_o + 1e-12)
    if mask is not None:
        spear_o = spearman_np(P_o[mask].ravel(), logE2_o[mask].ravel())
    else:
        spear_o = spearman_np(P_o.ravel(), logE2_o.ravel())
    z2_o = (T_o / (np.exp(P_o) + 1e-12))
    if mask is not None:
        z2o_mean = float(np.mean(z2_o[mask]))
        cov68_o = float(np.mean((z2_o <= thr68)[mask]))
        cov95_o = float(np.mean((z2_o <= thr95)[mask]))
    else:
        z2o_mean = float(np.mean(z2_o))
        cov68_o = float(np.mean((z2_o <= thr68)))
        cov95_o = float(np.mean((z2_o <= thr95)))
    metrics["overall"] = {
        "spearman": float(spear_o),
        "z2_mean": float(z2o_mean),
        "cov68": float(cov68_o),
        "cov95": float(cov95_o),
    }

spears_pre = [
    spearman_np(P_raw[...,j][mask].ravel(), logE2[...,j][mask].ravel()) if mask is not None
    else spearman_np(P_raw[...,j].ravel(), logE2[...,j].ravel())
    for j in range(3)
]
z2_pre = (T / (np.exp(P_raw) + 1e-12))
if mask is not None:
    m3p = np.repeat(mask[..., None], 3, axis=-1)
    z2p_mean = float(np.mean(z2_pre[m3p]))
    cov68_p = float(np.mean((z2_pre <= thr68)[m3p]))
    cov95_p = float(np.mean((z2_pre <= thr95)[m3p]))
else:
    z2p_mean = float(np.mean(z2_pre))
    cov68_p = float(np.mean((z2_pre <= thr68)))
    cov95_p = float(np.mean((z2_pre <= thr95)))
metrics_pre = {
    "spearman_axes": spears_pre,
    "spearman_mean": float(np.mean(spears_pre)),
    "z2_mean": z2p_mean,
    "cov68": cov68_p,
    "cov95": cov95_p,
}
if args.also_overall:
    var0 = np.exp(P_raw)
    var_o0 = np.mean(var0, axis=-1, keepdims=True)
    P_o0 = np.log(var_o0 + 1e-12)
    T_o0 = np.mean(T, axis=-1, keepdims=True)
    logE2_o0 = np.log(T_o0 + 1e-12)
    if mask is not None:
        spear_o0 = spearman_np(P_o0[mask].ravel(), logE2_o0[mask].ravel())
        z2_o0 = (T_o0 / (np.exp(P_o0) + 1e-12))
        z2o0_mean = float(np.mean(z2_o0[mask]))
        cov68_o0 = float(np.mean((z2_o0 <= thr68)[mask]))
        cov95_o0 = float(np.mean((z2_o0 <= thr95)[mask]))
    else:
        spear_o0 = spearman_np(P_o0.ravel(), logE2_o0.ravel())
        z2_o0 = (T_o0 / (np.exp(P_o0) + 1e-12))
        z2o0_mean = float(np.mean(z2_o0))
        cov68_o0 = float(np.mean((z2_o0 <= thr68)))
        cov95_o0 = float(np.mean((z2_o0 <= thr95)))
    metrics_pre["overall"] = {
        "spearman": float(spear_o0),
        "z2_mean": float(z2o0_mean),
        "cov68": float(cov68_o0),
        "cov95": float(cov95_o0),
    }
metrics["pre_calib"] = metrics_pre

print(json.dumps(metrics, indent=2))

if args.plots_dir:
    os.makedirs(args.plots_dir, exist_ok=True)
    for j,name in enumerate(["x","y","z"]):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,5))
        plt.scatter(logE2[...,j].ravel(), P[...,j].ravel(), s=4, alpha=0.3)
        mn = min(np.min(logE2[...,j]), np.min(P[...,j]))
        mx = max(np.max(logE2[...,j]), np.max(P[...,j]))
        plt.plot([mn,mx],[mn,mx],"r--",lw=2,alpha=0.5)
        plt.xlabel("GT log(e^2)"); plt.ylabel("Pred log(σ^2)")
        plt.title(f"Axis {name.upper()}  Spearman={spears[j]:.3f}")
        plt.grid(True, ls="--", alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, f"scatter_{name}.png"), dpi=180); plt.close()

        # Log–log calibration scatter: Pred variance vs Empirical squared error
        vx = np.exp(P[...,j].ravel()); vy = (T[...,j].ravel())
        z2j = (T[...,j] / (np.exp(P[...,j]) + 1e-12)).ravel()
        msk = (vx>1e-12) & (vy>1e-12)
        if mask is not None:
            msk = msk & (mask.ravel())
        vx = vx[msk]; vy = vy[msk]
        z2j = z2j[msk]
        # Rectangular removal in (variance,e2) space for calibr scatter
        if (args.rm_box_var_lo is not None and args.rm_box_var_hi is not None 
            and args.rm_box_e2_lo is not None and args.rm_box_e2_hi is not None):
            box = (vx >= float(args.rm_box_var_lo)) & (vx <= float(args.rm_box_var_hi)) \
                  & (vy >= float(args.rm_box_e2_lo)) & (vy <= float(args.rm_box_e2_hi))
            keep_box = ~box
            vx = vx[keep_box]; vy = vy[keep_box]; z2j = z2j[keep_box]
        # Elliptical removal aligned with y=x in log space (u,v axes)
        if (args.rm_ell_var_c is not None and args.rm_ell_e2_c is not None 
            and args.rm_ell_ru is not None and args.rm_ell_rv is not None and (vx.size>0)):
            lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
            lv0 = np.log(float(args.rm_ell_var_c) + 1e-12); le0 = np.log(float(args.rm_ell_e2_c) + 1e-12)
            u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
            u0 = (lv0 + le0)/np.sqrt(2.0); v0 = (le0 - lv0)/np.sqrt(2.0)
            ru = float(args.rm_ell_ru); rv = float(args.rm_ell_rv)
            inside = ((u - u0)/ru)**2 + ((v - v0)/rv)**2 <= 1.0
            keep_el = ~inside
            vx = vx[keep_el]; vy = vy[keep_el]; z2j = z2j[keep_el]
        shape_keep = None
        # shape-based region (log space)
        if args.shape_clip != "none" and vx.size > 0:
            lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
            if args.shape_clip == "ellipse":
                u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
                ulo, uhi = np.quantile(u, [1.0-args.shape_uq, args.shape_uq])
                ucen = 0.5*(ulo+uhi); ur = 0.5*(uhi-ulo)
                vmed = np.median(v); mad = 1.4826*np.median(np.abs(v - vmed)) + 1e-12
                shape_keep = (np.abs(u - ucen) <= ur) & (np.abs(v - vmed) <= args.shape_k*mad)
            else:
                qs = np.linspace(0.0, 1.0, int(max(args.shape_bins, 8))+1)
                xq = np.quantile(lx, qs)
                bin_idx = np.clip(np.digitize(lx, xq) - 1, 0, len(xq)-2)
                med = np.zeros(len(xq)-1); sc = np.zeros(len(xq)-1)
                for b in range(len(xq)-1):
                    sel = bin_idx == b
                    if np.any(sel):
                        m_ = np.median(ly[sel]); s_ = 1.4826*np.median(np.abs(ly[sel]-m_)) + 1e-12
                    else:
                        m_, s_ = 0.0, np.inf
                    med[b] = m_; sc[b] = s_
                dev = np.abs(ly - med[bin_idx]); thr = args.shape_k*sc[bin_idx]
                shape_keep = (dev <= thr)
        if args.shape_remove and (shape_keep is not None):
            vx = vx[shape_keep]; vy = vy[shape_keep]
            z2j = z2j[shape_keep]
            cols = None
        if args.resid_remove and z2j.size>0:
            r = np.log(z2j + 1e-12)
            med = np.median(r)
            mad = 1.4826*np.median(np.abs(r - med)) + 1e-12
            q = np.quantile(r, float(args.resid_q))
            keepR = r >= (q - float(args.resid_k)*mad)
            vx = vx[keepR]; vy = vy[keepR]; z2j = z2j[keepR]
        if args.plot_inliers_only:
            hi = float(getattr(args, 'inlier_hi', 3.0**2))
            lo = float(getattr(args, 'inlier_lo', 1.0/(3.0**2)))
            if 'inlier_thr' in args.__dict__ and args.__dict__.get('inlier_hi', None) is None:
                hi = float(args.inlier_thr)
            keep = (z2j >= lo) & (z2j <= hi)
            if (not args.shape_remove) and (shape_keep is not None):
                # only soft fade for points outside shape, keep all points
                lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
                soft = max(1e-6, min(0.49, float(args.shape_soft)))
                if args.shape_clip == "ellipse":
                    u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
                    ulo, uhi = np.quantile(u, [1.0-args.shape_uq, args.shape_uq])
                    ucen = 0.5*(ulo+uhi); ur = 0.5*(uhi-ulo)
                    vmed = np.median(v); mad = 1.4826*np.median(np.abs(v - vmed)) + 1e-12
                    ur_in = (1.0 - soft)*ur; ur_out = soft*ur + 1e-12
                    vr_in = (1.0 - soft)*args.shape_k*mad; vr_out = soft*args.shape_k*mad + 1e-12
                    du = np.abs(u - ucen); dv = np.abs(v - vmed)
                    wu = np.clip(1.0 - np.maximum(0.0, du - ur_in)/ur_out, 0.0, 1.0)
                    wv = np.clip(1.0 - np.maximum(0.0, dv - vr_in)/vr_out, 0.0, 1.0)
                    w = wu * wv
                else:
                    qs = np.linspace(0.0, 1.0, int(max(args.shape_bins, 8))+1)
                    xq = np.quantile(lx, qs)
                    bin_idx = np.clip(np.digitize(lx, xq) - 1, 0, len(xq)-2)
                    med = np.zeros(len(xq)-1); sc = np.zeros(len(xq)-1)
                    for b in range(len(xq)-1):
                        sel = bin_idx == b
                        if np.any(sel):
                            m_ = np.median(ly[sel]); s_ = 1.4826*np.median(np.abs(ly[sel]-m_)) + 1e-12
                        else:
                            m_, s_ = 0.0, np.inf
                        med[b] = m_; sc[b] = s_
                    dev = np.abs(ly - med[bin_idx]); thr = args.shape_k*sc[bin_idx]
                    thr_in = (1.0 - soft)*thr; thr_out = soft*thr + 1e-12
                    w = np.clip(1.0 - np.maximum(0.0, dev - thr_in)/thr_out, 0.0, 1.0)
                base = np.array([245/255, 166/255, 35/255, 1.0], dtype=np.float64)
                cols = np.tile(base, (vx.size,1))
                cols[:,3] = 0.06 + 0.42*w
            vx = vx[keep]; vy = vy[keep]
        # Optional shape-based soft fading (alpha weights)
        if 'cols' not in locals():
            cols = None
        # subsample for clarity if too many points
        if vx.size > 80000:
            idx = np.random.choice(vx.size, 80000, replace=False)
            vx = vx[idx]; vy = vy[idx]
        plt.figure(figsize=(8,5))
        if cols is None:
            plt.scatter(vx, vy, s=5, alpha=0.25, c="#f5a623")
        else:
            plt.scatter(vx, vy, s=5, c=cols, edgecolors='none')
        plt.xscale('log'); plt.yscale('log')
        mn = float(min(vx.min(), vy.min())); mx = float(max(vx.max(), vy.max()))
        plt.plot([mn,mx],[mn,mx], color="#0aa", lw=2, alpha=0.7)
        # quantile smoothing along x
        qs = np.linspace(0.02, 0.98, 48)
        xq = np.quantile(vx, qs)
        ymed = []
        for k in range(len(qs)-1):
            lo, hi = xq[k], xq[k+1]
            sel = (vx>=lo) & (vx<=hi)
            if sel.any():
                ymed.append(np.median(vy[sel]))
            else:
                ymed.append(np.nan)
        xmid = 0.5*(xq[:-1]+xq[1:]); ymed = np.array(ymed)
        good = ~np.isnan(ymed)
        if good.any():
            plt.plot(xmid[good], ymed[good], color="#2ec4b6", lw=2.0, alpha=0.9)
        plt.xlabel("Predicted variance σ²"); plt.ylabel("Empirical e²")
        plt.title(f"Calibration Scatter (log–log) • Axis {name.upper()}")
        plt.grid(True, which='both', ls='--', alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, f"calibr_scatter_{name}.png"), dpi=200); plt.close()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4)); plt.hist(z2.ravel(), bins=80, density=True)
    plt.axvline(1.0, color="r", ls="--", lw=2); plt.title("z^2 histogram (ideal mean≈1)")
    plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_hist.png"), dpi=180); plt.close()
    if args.also_overall:
        # Recompute overall for plotting convenience
        var = np.exp(P)
        var_o = np.mean(var, axis=-1, keepdims=True)
        P_o = np.log(var_o + 1e-12)
        T_o = np.mean(T, axis=-1, keepdims=True)
        logE2_o = np.log(T_o + 1e-12)
        # Pretty scatter: inliers vs outliers by z2 threshold, dashed perfect line, legend/metrics
        z2_o = (T_o / (np.exp(P_o) + 1e-12))
        hi = float(getattr(args, 'inlier_hi', 3.0**2))
        lo = float(getattr(args, 'inlier_lo', 0.0))
        # fallback to legacy single threshold if user set only inlier_thr
        if 'inlier_thr' in args.__dict__ and args.__dict__.get('inlier_thr', None) is not None and args.__dict__.get('inlier_hi', None) is None:
            hi = float(args.inlier_thr)
        x = logE2_o.ravel(); y = P_o.ravel(); z = z2_o.ravel()
        m = (z >= lo) & (z <= hi)
        base = mask.ravel() if (mask is not None) else np.ones_like(m, dtype=bool)
        # Optional shape-based removal in log space for overall
        if args.shape_clip != "none":
            lx = x.copy(); ly = y.copy()
            if args.shape_clip == "ellipse":
                u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
                ulo, uhi = np.quantile(u, [1.0-args.shape_uq, args.shape_uq])
                ucen = 0.5*(ulo+uhi); ur = 0.5*(uhi-ulo)
                vmed = np.median(v); mad = 1.4826*np.median(np.abs(v - vmed)) + 1e-12
                keep_shape = (np.abs(u - ucen) <= ur) & (np.abs(v - vmed) <= args.shape_k*mad)
            else:
                qs = np.linspace(0.0, 1.0, int(max(args.shape_bins, 8))+1)
                xq = np.quantile(lx, qs)
                bin_idx = np.clip(np.digitize(lx, xq) - 1, 0, len(xq)-2)
                med = np.zeros(len(xq)-1); sc = np.zeros(len(xq)-1)
                for b in range(len(xq)-1):
                    sel = bin_idx == b
                    if np.any(sel):
                        m_ = np.median(ly[sel]); s_ = 1.4826*np.median(np.abs(ly[sel]-m_)) + 1e-12
                    else:
                        m_, s_ = 0.0, np.inf
                    med[b] = m_; sc[b] = s_
                dev = np.abs(ly - med[bin_idx]); thr = args.shape_k*sc[bin_idx]
                keep_shape = (dev <= thr)
            if args.shape_remove:
                base = base & keep_shape
        if args.resid_remove:
            r = np.log(z2_o.ravel() + 1e-12)
            med = np.median(r)
            mad = 1.4826*np.median(np.abs(r - med)) + 1e-12
            q = np.quantile(r, float(args.resid_q))
            keepR = r >= (q - float(args.resid_k)*mad)
            base = base & keepR
        mn = float(min(x.min(), y.min())); mx = float(max(x.max(), y.max()))
        plt.figure(figsize=(7.5,6))
        if args.plot_inliers_only:
            idx = base & m
            plt.scatter(x[idx], y[idx], s=10, alpha=0.45, c="#1f77b4", label=f"Inlier ({idx.sum()})")
        else:
            # outliers先画，淡灰色（仅限掩码内）
            oidx = base & (~m)
            iidx = base & m
            plt.scatter(x[oidx], y[oidx], s=8, alpha=0.18, c="#9aa0a6", label=f"Outlier ({oidx.sum()})")
            plt.scatter(x[iidx], y[iidx], s=10, alpha=0.45, c="#1f77b4", label=f"Inlier ({iidx.sum()})")
        plt.plot([mn,mx],[mn,mx],"r--",lw=1.8,alpha=0.6,label="y=x")
        plt.xlabel("GT log(e^2) overall"); plt.ylabel("Pred log(σ^2) overall")
        # 标注核心指标
        title = (
            f"Overall Prediction  "
            f"(Spearman={metrics['overall']['spearman']:.3f}, "
            f"cov68={metrics['overall']['cov68']:.3f}, cov95={metrics['overall']['cov95']:.3f})"
        ) if "overall" in metrics else "Overall Prediction"
        plt.title(title)
        plt.grid(True, ls="--", alpha=0.25)
        plt.legend(frameon=False, loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, "scatter_overall.png"), dpi=200); plt.close()

        z2_o = (T_o / (np.exp(P_o) + 1e-12))
        plt.figure(figsize=(6,4)); plt.hist(z2_o.ravel(), bins=80, density=True)
        plt.axvline(1.0, color="r", ls="--", lw=2); plt.title("z^2 overall (ideal mean≈1)")
        plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_hist_overall.png"), dpi=180); plt.close()
        # Overall log–log calibration scatter
        vx = np.exp(P_o.ravel()); vy = T_o.ravel()
        z2o = (T_o / (np.exp(P_o) + 1e-12)).ravel()
        msk = (vx>1e-12) & (vy>1e-12)
        if args.apply_mask:
            msk = msk & (mask.ravel())
        vx = vx[msk]; vy = vy[msk]
        z2o = z2o[msk]
        # Rectangular removal in (variance,e2) space for calibr scatter overall
        if (args.rm_box_var_lo is not None and args.rm_box_var_hi is not None 
            and args.rm_box_e2_lo is not None and args.rm_box_e2_hi is not None):
            box = (vx >= float(args.rm_box_var_lo)) & (vx <= float(args.rm_box_var_hi)) \
                  & (vy >= float(args.rm_box_e2_lo)) & (vy <= float(args.rm_box_e2_hi))
            keep_box = ~box
            vx = vx[keep_box]; vy = vy[keep_box]; z2o = z2o[keep_box]
        # Elliptical removal aligned with y=x in log space (u,v axes)
        if (args.rm_ell_var_c is not None and args.rm_ell_e2_c is not None 
            and args.rm_ell_ru is not None and args.rm_ell_rv is not None and (vx.size>0)):
            lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
            lv0 = np.log(float(args.rm_ell_var_c) + 1e-12); le0 = np.log(float(args.rm_ell_e2_c) + 1e-12)
            u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
            u0 = (lv0 + le0)/np.sqrt(2.0); v0 = (le0 - lv0)/np.sqrt(2.0)
            ru = float(args.rm_ell_ru); rv = float(args.rm_ell_rv)
            inside = ((u - u0)/ru)**2 + ((v - v0)/rv)**2 <= 1.0
            keep_el = ~inside
            vx = vx[keep_el]; vy = vy[keep_el]; z2o = z2o[keep_el]
        # shape-based region in log space
        shape_keep_o = None
        if args.shape_clip != "none" and vx.size>0:
            lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
            if args.shape_clip == "ellipse":
                u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
                ulo, uhi = np.quantile(u, [1.0-args.shape_uq, args.shape_uq])
                ucen = 0.5*(ulo+uhi); ur = 0.5*(uhi-ulo)
                vmed = np.median(v); mad = 1.4826*np.median(np.abs(v - vmed)) + 1e-12
                shape_keep_o = (np.abs(u - ucen) <= ur) & (np.abs(v - vmed) <= args.shape_k*mad)
            else:
                qs = np.linspace(0.0, 1.0, int(max(args.shape_bins, 8))+1)
                xq = np.quantile(lx, qs)
                bin_idx = np.clip(np.digitize(lx, xq) - 1, 0, len(xq)-2)
                med = np.zeros(len(xq)-1); sc = np.zeros(len(xq)-1)
                for b in range(len(xq)-1):
                    sel = bin_idx == b
                    if np.any(sel):
                        m_ = np.median(ly[sel]); s_ = 1.4826*np.median(np.abs(ly[sel]-m_)) + 1e-12
                    else:
                        m_, s_ = 0.0, np.inf
                    med[b] = m_; sc[b] = s_
                dev = np.abs(ly - med[bin_idx]); thr = args.shape_k*sc[bin_idx]
                shape_keep_o = (dev <= thr)
        if args.shape_remove and (shape_keep_o is not None):
            vx = vx[shape_keep_o]; vy = vy[shape_keep_o]; z2o = z2o[shape_keep_o]
            cols = None
        # residual-based removal on log z^2
        if args.resid_remove and z2o.size>0:
            r = np.log(z2o + 1e-12)
            medr = np.median(r)
            madr = 1.4826*np.median(np.abs(r - medr)) + 1e-12
            qr = np.quantile(r, float(args.resid_q))
            keepR = r >= (qr - float(args.resid_k)*madr)
            vx = vx[keepR]; vy = vy[keepR]; z2o = z2o[keepR]
        if args.plot_inliers_only:
            hi = float(getattr(args, 'inlier_hi', 3.0**2))
            lo = float(getattr(args, 'inlier_lo', 0.0))
            if 'inlier_thr' in args.__dict__ and args.__dict__.get('inlier_hi', None) is None:
                hi = float(args.inlier_thr)
            keep = (z2o >= lo) & (z2o <= hi)
            # soft fade if not hard removed by shape
            if (not args.shape_remove) and (shape_keep_o is not None):
                lx = np.log(vx + 1e-12); ly = np.log(vy + 1e-12)
                soft = max(1e-6, min(0.49, float(args.shape_soft)))
                if args.shape_clip == "ellipse":
                    u = (lx + ly)/np.sqrt(2.0); v = (ly - lx)/np.sqrt(2.0)
                    ulo, uhi = np.quantile(u, [1.0-args.shape_uq, args.shape_uq])
                    ucen = 0.5*(ulo+uhi); ur = 0.5*(uhi-ulo)
                    vmed = np.median(v); mad = 1.4826*np.median(np.abs(v - vmed)) + 1e-12
                    ur_in = (1.0 - soft)*ur; ur_out = soft*ur + 1e-12
                    vr_in = (1.0 - soft)*args.shape_k*mad; vr_out = soft*args.shape_k*mad + 1e-12
                    du = np.abs(u - ucen); dv = np.abs(v - vmed)
                    wu = np.clip(1.0 - np.maximum(0.0, du - ur_in)/ur_out, 0.0, 1.0)
                    wv = np.clip(1.0 - np.maximum(0.0, dv - vr_in)/vr_out, 0.0, 1.0)
                    w = wu * wv
                else:
                    qs = np.linspace(0.0, 1.0, int(max(args.shape_bins, 8))+1)
                    xq = np.quantile(lx, qs)
                    bin_idx = np.clip(np.digitize(lx, xq) - 1, 0, len(xq)-2)
                    med = np.zeros(len(xq)-1); sc = np.zeros(len(xq)-1)
                    for b in range(len(xq)-1):
                        sel = bin_idx == b
                        if np.any(sel):
                            m_ = np.median(ly[sel]); s_ = 1.4826*np.median(np.abs(ly[sel]-m_)) + 1e-12
                        else:
                            m_, s_ = 0.0, np.inf
                        med[b] = m_; sc[b] = s_
                    dev = np.abs(ly - med[bin_idx]); thr = args.shape_k*sc[bin_idx]
                    thr_in = (1.0 - soft)*thr; thr_out = soft*thr + 1e-12
                    w = np.clip(1.0 - np.maximum(0.0, dev - thr_in)/thr_out, 0.0, 1.0)
                base = np.array([245/255, 166/255, 35/255, 1.0], dtype=np.float64)
                cols = np.tile(base, (vx.size,1))
                cols[:,3] = 0.06 + 0.42*w
            vx = vx[keep]; vy = vy[keep]
        # Optional shape-based soft fading (alpha weights)
        if 'cols' not in locals():
            cols = None
        if vx.size > 80000:
            idx = np.random.choice(vx.size, 80000, replace=False)
            vx = vx[idx]; vy = vy[idx]
        plt.figure(figsize=(8,5))
        plt.scatter(vx, vy, s=5, alpha=0.25, c="#f5a623")
        plt.xscale('log'); plt.yscale('log')
        mn = float(min(vx.min(), vy.min())); mx = float(max(vx.max(), vy.max()))
        plt.plot([mn,mx],[mn,mx], color="#0aa", lw=2, alpha=0.7)
        qs = np.linspace(0.02, 0.98, 48)
        xq = np.quantile(vx, qs)
        ymed = []
        for k in range(len(qs)-1):
            lo, hi = xq[k], xq[k+1]
            sel = (vx>=lo) & (vx<=hi)
            if sel.any():
                ymed.append(np.median(vy[sel]))
            else:
                ymed.append(np.nan)
        xmid = 0.5*(xq[:-1]+xq[1:]); ymed = np.array(ymed)
        good = ~np.isnan(ymed)
        if good.any():
            plt.plot(xmid[good], ymed[good], color="#2ec4b6", lw=2.0, alpha=0.9)
        plt.xlabel("Predicted variance σ²"); plt.ylabel("Empirical e²")
        plt.title("Calibration Scatter (log–log) • Overall")
        plt.grid(True, which='both', ls='--', alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(args.plots_dir, "calibr_scatter_overall.png"), dpi=200); plt.close()
    if args.compare_uncalibrated and args.calib_json:
        z2_pre = (T / (np.exp(P_raw) + 1e-12))
        plt.figure(figsize=(6,4))
        plt.hist(z2_pre.ravel(), bins=80, density=True, alpha=0.4, label="pre")
        plt.hist(z2.ravel(), bins=80, density=True, alpha=0.4, label="post")
        plt.axvline(thr68, color="g", ls="--", lw=1)
        plt.axvline(thr95, color="g", ls=":", lw=1)
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_hist_compare.png"), dpi=180); plt.close()
        # Coverage curve
        ts = np.quantile(np.concatenate([z2_pre.ravel(), z2.ravel()]), np.linspace(0.01,0.99,99))
        cov_pre = [(z2_pre<=t).mean() for t in ts]
        cov_post = [(z2<=t).mean() for t in ts]
        plt.figure(figsize=(6,4))
        plt.plot(ts, cov_pre, label="pre")
        plt.plot(ts, cov_post, label="post")
        plt.axvline(thr68, color="g", ls="--", lw=1)
        plt.axvline(thr95, color="g", ls=":", lw=1)
        plt.xlabel("threshold t"); plt.ylabel("coverage P(z^2<=t)")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_cov_curve.png"), dpi=180); plt.close()
        # Overlay hist of distributions: Pred log variance vs GT log error^2 (axes)
        for j,name in enumerate(["x","y","z"]):
            plt.figure(figsize=(6,4))
            plt.hist(P_raw[...,j].ravel(), bins=120, density=True, alpha=0.45, label="Pred pre")
            plt.hist(P[...,j].ravel(),     bins=120, density=True, alpha=0.45, label="Pred post")
            plt.hist(logE2[...,j].ravel(), bins=120, density=True, alpha=0.35, label="GT log(e^2)")
            plt.xlabel("value (log variance / log error^2)"); plt.ylabel("Density")
            plt.title(f"Axis {name.upper()} distributions")
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, f"dist_logvar_compare_{name}.png"), dpi=180); plt.close()
        if args.also_overall:
            var0 = np.exp(P_raw); var1 = np.exp(P)
            P_o0 = np.log(np.mean(var0, axis=-1, keepdims=True) + 1e-12)
            P_o1 = np.log(np.mean(var1, axis=-1, keepdims=True) + 1e-12)
            T_o = np.mean(T, axis=-1, keepdims=True)
            z2_o_pre = (T_o / (np.exp(P_o0) + 1e-12))
            z2_o_post = (T_o / (np.exp(P_o1) + 1e-12))
            plt.figure(figsize=(6,4))
            plt.hist(z2_o_pre.ravel(), bins=80, density=True, alpha=0.4, label="pre")
            plt.hist(z2_o_post.ravel(), bins=80, density=True, alpha=0.4, label="post")
            plt.axvline(thr68, color="g", ls="--", lw=1)
            plt.axvline(thr95, color="g", ls=":", lw=1)
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_hist_overall_compare.png"), dpi=180); plt.close()
            ts2 = np.quantile(np.concatenate([z2_o_pre.ravel(), z2_o_post.ravel()]), np.linspace(0.01,0.99,99))
            cov_o_pre = [(z2_o_pre<=t).mean() for t in ts2]
            cov_o_post = [(z2_o_post<=t).mean() for t in ts2]
            plt.figure(figsize=(6,4))
            plt.plot(ts2, cov_o_pre, label="pre")
            plt.plot(ts2, cov_o_post, label="post")
            plt.axvline(thr68, color="g", ls="--", lw=1)
            plt.axvline(thr95, color="g", ls=":", lw=1)
            plt.xlabel("threshold t"); plt.ylabel("coverage P(z^2<=t)")
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "z2_cov_curve_overall.png"), dpi=180); plt.close()
            # Overlay hist of overall distributions (Pred pre/post vs GT)
            logE2_o = np.log(T_o + 1e-12)
            plt.figure(figsize=(6,4))
            plt.hist(P_o0.ravel(),    bins=120, density=True, alpha=0.45, label="Pred pre")
            plt.hist(P_o1.ravel(),    bins=120, density=True, alpha=0.45, label="Pred post")
            plt.hist(logE2_o.ravel(), bins=120, density=True, alpha=0.35, label="GT log(e^2)")
            plt.xlabel("value (log variance / log error^2)"); plt.ylabel("Density")
            plt.title("Overall distributions")
            plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(args.plots_dir, "dist_logvar_compare_overall.png"), dpi=180); plt.close()
    print("Plots saved to", args.plots_dir)

if args.save_pred_npz:
    try:
        var = np.exp(P)
        var_o = np.mean(var, axis=-1)
        np.savez_compressed(args.save_pred_npz, P_axes=P, P_overall=np.log(var_o + 1e-12))
    except Exception as e:
        print("Failed to save predictions:", e)
