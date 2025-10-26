import argparse, os, glob, json
import numpy as np

KEYS = {
    "acc": ("E2_IMU_ACC", "Y_IMU_ACC"),
    "gyr": ("E2_IMU_GYR", "Y_IMU_GYR"),
}

def _pick_key(d, route):
    for k in KEYS[route]:
        if k in d.files:
            return k
    return None

def _finite(a: np.ndarray):
    return np.isfinite(a)

def _overall_mean_e2(E2: np.ndarray) -> np.ndarray:
    # mean across 3 axes with finite-mask
    m = _finite(E2)
    num = np.where(m, E2, 0.0).sum(axis=-1)
    den = m.sum(axis=-1)
    den = np.where(den <= 0, 1.0, den)
    return num / den

def _summary(v: np.ndarray) -> dict:
    v = v[np.isfinite(v)]
    out = {
        "count": int(v.size),
    }
    if v.size:
        qs = [1, 10, 25, 50, 75, 90, 99]
        out.update({
            "min": float(np.min(v)),
            **{f"p{q}": float(np.percentile(v, q)) for q in qs},
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
        })
    else:
        for k in ["min","p1","p10","p25","p50","p75","p90","p99","max","mean","std"]:
            out[k] = None
    return out

def _diff(a: dict, b: dict, keys):
    # return b - a for selected keys
    d = {}
    for k in keys:
        va, vb = a.get(k, None), b.get(k, None)
        if va is None or vb is None:
            d[k] = None
        else:
            d[k] = float(vb - va)
    return d

def rec_range_from_train(log_over_train: np.ndarray):
    v = log_over_train[np.isfinite(log_over_train)]
    if v.size == 0:
        return None, None
    lo = float(np.percentile(v, 1))
    hi = float(np.percentile(v, 99))
    return round(lo - 0.5, 2), round(hi + 0.5, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--route", choices=["acc","gyr"], required=True)
    ap.add_argument("--test_pattern", default="MH_03")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))
    if not files:
        raise SystemExit(f"No NPZ files in folder: {args.folder}")

    log_over_train = []
    log_over_test = []
    per_seq = []

    for fp in files:
        d = np.load(fp, allow_pickle=True)
        k = _pick_key(d, args.route)
        if k is None:
            continue
        E2 = d[k]
        if E2.ndim == 2:
            E2 = E2[..., None]
        if E2.shape[-1] == 1:
            E2 = np.repeat(E2, 3, axis=-1)
        E2 = E2.astype(np.float64)
        over = _overall_mean_e2(E2)
        log_over = np.log(np.maximum(over, 0.0) + 1e-12).ravel()
        nm = os.path.basename(fp)
        stats_seq = _summary(log_over)
        per_seq.append({"file": nm, "logE2_overall": stats_seq})
        if args.test_pattern and args.test_pattern in nm:
            log_over_test.append(log_over)
        else:
            log_over_train.append(log_over)

    def cat(xs):
        return np.concatenate(xs, axis=0) if xs else np.array([], dtype=np.float64)

    log_tr = cat(log_over_train)
    log_te = cat(log_over_test)

    summ_tr = _summary(log_tr)
    summ_te = _summary(log_te)
    keys = ["p1","p10","p25","p50","p75","p90","p99","mean","std"]
    diffs = _diff(summ_tr, summ_te, keys)

    rec_lo, rec_hi = rec_range_from_train(log_tr)

    out = {
        "route": args.route,
        "folder": args.folder,
        "test_pattern": args.test_pattern,
        "train": {"logE2_overall": summ_tr},
        "test":  {"logE2_overall": summ_te},
        "diff":  {"train_to_test_logE2_overall": diffs},
        "recommend": {"logv_min": rec_lo, "logv_max": rec_hi},
        "per_seq_examples": per_seq[:5],
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved", args.out_json)

if __name__ == "__main__":
    main()
