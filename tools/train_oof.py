
#!/usr/bin/env python3
import argparse, json, os, numpy as np, torch
from torch.utils.data import DataLoader
import torch.optim as optim
from imu.datasets_imu_frames import IMUFrames
from imu.model_sa3 import IMUSA3
from imu.losses import nll_student_t_diag

def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def spearman_np(x,y):
    rx = x.argsort().argsort().astype(np.float64); ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    den = (rx**2).sum()**0.5 * (ry**2).sum()**0.5 + 1e-12
    return float((rx*ry).sum() / den)

def run_one_fold(args, fold_id, tr_ids, va_ids, save_dir):
    dev = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    os.makedirs(save_dir, exist_ok=True)

    full = IMUFrames(args.npz, route=args.route, scaler_npz=None)
    seq = full.seq
    tr_idx = np.where(np.isin(seq, tr_ids))[0]; va_idx = np.where(np.isin(seq, va_ids))[0]

    ds_tr = IMUFrames(args.npz, route=args.route, scaler_npz=None, subset_ids=tr_idx)
    scaler_path = os.path.join(save_dir, "scaler.npz")
    np.savez(scaler_path, mean=ds_tr.scaler["mean"], std=ds_tr.scaler["std"])
    ds_va = IMUFrames(args.npz, route=args.route, scaler_npz=scaler_path, subset_ids=va_idx)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)

    d_in = ds_tr.X.shape[-1]
    mdl = IMUSA3(d_in=d_in, d_model=args.d_model, n_tcn=args.n_tcn, k=args.kernel, n_tf=args.n_tf,
                 logv_min=args.logv_min, logv_max=args.logv_max).to(dev)
    opt = optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.scheduler == "plateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    elif args.scheduler == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    else:
        sched = None

    best = {"val_obj": 1e9, "ep": -1}
    def _feat_stats(t):
        return float(torch.nan_to_num(t).mean().item()), float(torch.nan_to_num(t).std().item()), float(torch.nan_to_num(t).min().item()), float(torch.nan_to_num(t).max().item())

    epochs_no_improve = 0
    for ep in range(1, args.epochs+1):
        mdl.train(); tot=0.0; n=0; vtr_sum=0.0; vtr_n=0
        first_batch = True
        dbg_count = 0
        for b in dl_tr:
            X = b["X"].to(dev); E = b["E"].to(dev); M = b["M"].to(dev)
            if args.ignore_mask:
                M = torch.ones_like(M)
            logv, _ = mdl(X)
            loss = nll_student_t_diag(logv, E, M, nu=args.nu)
            vf_now = ((torch.isfinite(logv).all(dim=-1) & torch.isfinite(E).all(dim=-1) & (M>0.5)).float().mean().item())
            if not torch.isfinite(loss) or vf_now < 0.95:
                opt.zero_grad()
                if args.debug:
                    print(json.dumps({"ep": ep, "phase": "train", "skip_batch": True, "vf": float(vf_now), "finite_loss": bool(torch.isfinite(loss))}))
                continue
            opt.zero_grad(); loss.backward();
            pre_gn = 0.0
            for p in mdl.parameters():
                if p.grad is not None:
                    pre_gn += float(torch.norm(p.grad.detach(), p=2).item()**2)
            pre_gn = pre_gn ** 0.5
            if args.grad_clip > 0:
                post_gn = float(torch.nn.utils.clip_grad_norm_(mdl.parameters(), args.grad_clip).item())
            else:
                post_gn = pre_gn
            opt.step()
            tot += float(loss.item()); n += 1
            vf = ((torch.isfinite(logv).all(dim=-1) & torch.isfinite(E).all(dim=-1) & (M>0.5)).float().mean().item())
            if first_batch:
                mmean = float((M>0.5).float().mean().item())
                first_batch = False
            vtr_sum += vf; vtr_n += 1

            if args.debug and dbg_count < 3 and (ep == 1 or vf < 0.01):
                with torch.no_grad():
                    h = mdl.enc(X.transpose(1,2)); h = mdl.tcn(h)
                    if mdl.tf is not None:
                        h = mdl.tf(h.transpose(1,2)).transpose(1,2)
                    s_raw = mdl.head_s(h).transpose(1,2)
                    a_raw = mdl.head_a(h).transpose(1,2)
                    s_b = mdl.bound(s_raw)
                    a_c = a_raw - a_raw.mean(dim=-1, keepdim=True)
                    logv_dbg = mdl.bound(s_b + a_c)
                    ms = _feat_stats(s_raw)
                    ma = _feat_stats(a_raw)
                    ml = _feat_stats(logv_dbg)
                    mx = _feat_stats(X)
                    me = _feat_stats(E)
                    print(json.dumps({
                        "ep": ep, "phase": "train", "b": dbg_count,
                        "vf": float(vf), "mmean": float(mmean),
                        "pre_gn": float(pre_gn), "post_gn": float(post_gn),
                        "s_raw": {"mean": ms[0], "std": ms[1], "min": ms[2], "max": ms[3]},
                        "a_raw": {"mean": ma[0], "std": ma[1], "min": ma[2], "max": ma[3]},
                        "logv":  {"mean": ml[0], "std": ml[1], "min": ml[2], "max": ml[3]},
                        "X":      {"mean": mx[0], "std": mx[1], "min": mx[2], "max": mx[3]},
                        "E":      {"mean": me[0], "std": me[1], "min": me[2], "max": me[3]},
                        "finite_logv_frac": float(torch.isfinite(logv_dbg).all(dim=-1).float().mean().item()),
                        "finite_E_frac": float(torch.isfinite(E).all(dim=-1).float().mean().item()),
                    }))
                    dbg_count += 1
        tr_loss = tot/max(n,1)

        mdl.eval()
        with torch.no_grad():
            vs=0.0; vn=0; vva_sum=0.0; vva_n=0
            preds=[]; gts=[]
            dbg_count_va = 0
            for b in dl_va:
                X = b["X"].to(dev); E=b["E"].to(dev); M=b["M"].to(dev)
                if args.ignore_mask:
                    M = torch.ones_like(M)
                logv,_ = mdl(X)
                loss = nll_student_t_diag(logv, E, M, nu=args.nu)
                vs += float(loss.item()); vn += 1
                vf = ((torch.isfinite(logv).all(dim=-1) & torch.isfinite(E).all(dim=-1) & (M>0.5)).float().mean().item())
                vva_sum += vf; vva_n += 1
                preds.append(logv.cpu().numpy()); gts.append((E.cpu().numpy()**2))
                if args.debug and dbg_count_va < 2 and (ep == 1 or vf < 0.01):
                    s_raw = mdl.head_s(mdl.tcn(mdl.enc(X.transpose(1,2)))).transpose(1,2)
                    a_raw = mdl.head_a(mdl.tcn(mdl.enc(X.transpose(1,2)))).transpose(1,2)
                    s_b = mdl.bound(s_raw); a_c = a_raw - a_raw.mean(dim=-1, keepdim=True)
                    logv_dbg = mdl.bound(s_b + a_c)
                    ms = _feat_stats(s_raw); ma = _feat_stats(a_raw); ml = _feat_stats(logv_dbg)
                    print(json.dumps({
                        "ep": ep, "phase": "val", "b": dbg_count_va,
                        "vf": float(vf),
                        "s_raw": {"mean": ms[0], "std": ms[1], "min": ms[2], "max": ms[3]},
                        "a_raw": {"mean": ma[0], "std": ma[1], "min": ma[2], "max": ma[3]},
                        "logv":  {"mean": ml[0], "std": ml[1], "min": ml[2], "max": ml[3]},
                        "finite_logv_frac": float(torch.isfinite(logv_dbg).all(dim=-1).float().mean().item()),
                        "finite_E_frac": float(torch.isfinite(E).all(dim=-1).float().mean().item()),
                    }))
                    dbg_count_va += 1
            va_loss = vs/max(vn,1)
        P = np.concatenate(preds,0); T = np.concatenate(gts,0)
        spear = np.mean([spearman_np(P[...,j].ravel(), np.log(T[...,j].ravel()+1e-12)) for j in range(3)])
        z2 = (T / (np.exp(P) + 1e-12))
        z2_mean = float(np.mean(z2))
        cov68 = float(np.mean((z2 <= 1.046**2)))
        var = np.exp(P)
        var_o = np.mean(var, axis=-1, keepdims=True)
        P_o = np.log(var_o + 1e-12)
        T_o = np.mean(T, axis=-1, keepdims=True)
        z2_o = (T_o / (np.exp(P_o) + 1e-12))
        z2o_mean = float(np.mean(z2_o))
        cov68_o = float(np.mean((z2_o <= 1.046**2)))
        obj = va_loss - 0.1*spear
        vtr = vtr_sum/max(vtr_n,1); vva = vva_sum/max(vva_n,1)
        if 'mmean' in locals():
            print(f"[fold {fold_id}] ep {ep:03d} | tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} spear={spear:.3f} obj={obj:.4f} vtr={vtr:.3f} vva={vva:.3f} mtr={mmean:.3f} z2={z2_mean:.3f} cov={cov68:.3f} oz2={z2o_mean:.3f} ocov={cov68_o:.3f}")
        else:
            print(f"[fold {fold_id}] ep {ep:03d} | tr_loss={tr_loss:.4f} val_loss={va_loss:.4f} spear={spear:.3f} obj={obj:.4f} vtr={vtr:.3f} vva={vva:.3f} z2={z2_mean:.3f} cov={cov68:.3f} oz2={z2o_mean:.3f} ocov={cov68_o:.3f}")
        improved = False
        if obj < best["val_obj"] - args.early_stop_delta:
            best.update({"val_obj":obj,"ep":ep})
            torch.save({"model": mdl.state_dict(), "d_in": d_in, "args": vars(args)}, os.path.join(save_dir,"best.pt"))
            improved = True
        if sched is not None:
            if args.scheduler == "plateau":
                sched.step(obj)
            else:
                sched.step()
        if improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            break

    return os.path.join(save_dir,"best.pt"), scaler_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--kfold_json", required=True)
    ap.add_argument("--save_root", required=True)
    ap.add_argument("--geom_stats", default=None)  # parity with VIS
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--nu", type=float, default=6.0)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_tcn", type=int, default=4)
    ap.add_argument("--n_tf", type=int, default=0)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--logv_min", type=float, default=-10)
    ap.add_argument("--logv_max", type=float, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ignore_mask", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--early_stop_patience", type=int, default=15)
    ap.add_argument("--early_stop_delta", type=float, default=0.0)
    ap.add_argument("--scheduler", choices=["none","plateau","cosine"], default="plateau")
    args = ap.parse_args()

    set_seed(42)
    with open(args.kfold_json,"r",encoding="utf-8") as f:
        KF = json.load(f)["folds"]

    os.makedirs(args.save_root, exist_ok=True)

    all_pred=[]; all_gt=[]; all_mask=[]
    base_ds = IMUFrames(args.npz, route=args.route, scaler_npz=None)
    base_seq = base_ds.seq

    for i,fold in enumerate(KF):
        fold_dir = os.path.join(args.save_root, f"fold{i}")
        best, scaler = run_one_fold(args, i, np.array(fold["train"]), np.array(fold["val"]), fold_dir)

        va_idx = np.where(np.isin(base_seq, fold["val"]))[0]
        ds_va = IMUFrames(args.npz, route=args.route, scaler_npz=scaler, subset_ids=va_idx)
        dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
        ckpt = torch.load(best, map_location="cpu")
        mdl = IMUSA3(d_in=ckpt["d_in"], d_model=ckpt["args"]["d_model"], n_tcn=ckpt["args"]["n_tcn"],
                     k=ckpt["args"]["kernel"], n_tf=ckpt["args"]["n_tf"],
                     logv_min=ckpt["args"]["logv_min"], logv_max=ckpt["args"]["logv_max"])
        mdl.load_state_dict(ckpt["model"]); mdl.eval()
        with torch.no_grad():
            preds=[]; gts=[]; masks=[]
            for b in dl_va:
                logv,_ = mdl(b["X"])
                preds.append(logv.numpy())
                gts.append((b["E"].numpy()**2))
                masks.append(b["M"].numpy())
        all_pred.append(np.concatenate(preds,0)); all_gt.append(np.concatenate(gts,0)); all_mask.append(np.concatenate(masks,0))

    P = np.concatenate(all_pred,0); T = np.concatenate(all_gt,0)

if __name__ == "__main__":
    main()
