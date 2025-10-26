import argparse, os, glob, json
import numpy as np

KEYS = {
    "acc": ("E2_IMU_ACC", "Y_IMU_ACC"),
    "gyr": ("E2_IMU_GYR", "Y_IMU_GYR"),
}

def pick_key(d, route):
    for k in KEYS[route]:
        if k in d.files:
            return k
    return None

def _finite(a: np.ndarray):
    return np.isfinite(a)

def agg_stats(a: np.ndarray):
    a = a.astype(np.float64)
    m = _finite(a)
    valid = a[m]
    out = {
        "count_total": int(a.size),
        "count_valid": int(valid.size),
        "count_invalid": int(a.size - valid.size),
        "frac_valid": float(valid.size / max(1, a.size)),
    }
    if valid.size:
        out.update({
            "min": float(np.min(valid)),
            "p1": float(np.percentile(valid, 1)),
            "p50": float(np.percentile(valid, 50)),
            "p99": float(np.percentile(valid, 99)),
            "max": float(np.max(valid)),
        })
    else:
        out.update({k: None for k in ["min","p1","p50","p99","max"]})
    return out

def scan_folder(folder: str, route: str, test_pattern: str):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if not files:
        return {"error": f"no npz in {folder}"}

    per_seq = []
    train_E2 = []; test_E2 = []
    train_log = []; test_log = []
    train_log_over = []; test_log_over = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        k = pick_key(d, route)
        if k is None:
            continue
        E2 = d[k].astype(np.float64)
        # valid values only
        vmask = _finite(E2)
        E2v = E2[vmask]
        # per-step overall mean across 3 axes with finite-mask
        m = _finite(E2)
        num = np.where(m, E2, 0.0).sum(axis=-1)
        den = m.sum(axis=-1)
        den = np.where(den <= 0, 1.0, den)
        meanE2 = num / den
        logE2 = np.log(np.maximum(E2v, 0.0) + 1e-12)
        logE2_over = np.log(np.maximum(meanE2, 0.0) + 1e-12)
        nm = os.path.basename(fp)
        per_seq.append({
            "file": nm,
            "E2": agg_stats(E2),
            "logE2": agg_stats(logE2),
            "logE2_overall": agg_stats(logE2_over),
        })
        if test_pattern and test_pattern in nm:
            test_E2.append(E2); test_log.append(logE2); test_log_over.append(logE2_over)
        else:
            train_E2.append(E2); train_log.append(logE2); train_log_over.append(logE2_over)

    def cat(xs):
        return np.concatenate(xs, axis=0) if xs else np.array([0.0])

    A_all = cat(train_E2 + test_E2)
    L_all = cat(train_log + test_log)
    LO_all = cat(train_log_over + test_log_over)

    A_tr = cat(train_E2); L_tr = cat(train_log); LO_tr = cat(train_log_over)
    A_te = cat(test_E2);  L_te = cat(test_log);  LO_te = cat(test_log_over)

    # heuristic: pick tight but safe box from overall log E2 (valid only)
    def rec_range(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None, None
        lo = float(np.percentile(v, 1))
        hi = float(np.percentile(v, 99))
        return round(lo - 0.5, 2), round(hi + 0.5, 2)

    rec_lo, rec_hi = rec_range(LO_tr if LO_tr.size else LO_all)

    return {
        "route": route,
        "folder": folder,
        "test_pattern": test_pattern,
        "per_seq_examples": per_seq[:3],
        "all": {"E2": agg_stats(A_all), "logE2": agg_stats(L_all), "logE2_overall": agg_stats(LO_all)},
        "train": {"E2": agg_stats(A_tr), "logE2": agg_stats(L_tr), "logE2_overall": agg_stats(LO_tr)} if A_tr.size else {},
        "test": {"E2": agg_stats(A_te), "logE2": agg_stats(L_te), "logE2_overall": agg_stats(LO_te)} if A_te.size else {},
        "recommend": {"logv_min": rec_lo, "logv_max": rec_hi}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--test_pattern", default="MH_03")
    args = ap.parse_args()

    res = scan_folder(args.folder, args.route, args.test_pattern)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
