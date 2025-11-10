#!/usr/bin/env python3
import argparse, json, numpy as np, os

def pick(d, keys):
    for k in keys:
        if k in d.files:
            return d[k]
    return None

def stats(arr, mask=None):
    if mask is not None:
        arr = arr[mask]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": None, "max": None, "q": []}
    qs = [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q": [float(np.quantile(arr, q)) for q in qs]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--route', choices=['acc','gyr'], required=True)
    ap.add_argument('--apply_mask', action='store_true')
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    files = list(d.files)

    if args.route == 'acc':
        E2 = pick(d, ['E2_IMU_ACC', 'Y_IMU_ACC', 'E2_ACC', 'E2'])
    else:
        E2 = pick(d, ['E2_IMU_GYR', 'Y_IMU_GYR', 'E2_GYR', 'E2'])

    if E2 is None:
        # fallback: try E then square
        E = pick(d, ['E_IMU_ACC','E_IMU_GYR','E_acc','E_gyr','E'])
        if E is not None:
            E2 = (E.astype(np.float64) ** 2)
        else:
            print(json.dumps({"error":"no E2/E found in NPZ","files":files}, ensure_ascii=False, indent=2))
            return

    E2 = E2.astype(np.float64)
    if E2.ndim == 2:
        E2 = E2[..., None]
    if E2.shape[-1] == 1:
        E2 = np.repeat(E2, 3, axis=-1)

    # optional mask
    M = pick(d, ['MASK_IMU','mask'])
    mask = None
    if args.apply_mask and (M is not None):
        M = M.astype(np.float64)
        mask = (M > 0.5)
        if mask.ndim == 2:
            mask = np.repeat(mask[..., None], 3, axis=-1)

    logE2 = np.log(E2 + 1e-12)

    out = {
        'files': files,
        'shape_E2': tuple(int(x) for x in E2.shape),
        'mask_present': bool(M is not None),
        'mask_applied': bool(mask is not None),
        'e2_axes': [stats(E2[...,i], None if mask is None else mask[...,i]) for i in range(E2.shape[-1])],
        'loge2_axes': [stats(logE2[...,i], None if mask is None else mask[...,i]) for i in range(E2.shape[-1])],
        'e2_overall': stats(E2, mask),
        'loge2_overall': stats(logE2, mask),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
