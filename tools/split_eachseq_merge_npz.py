#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-sequence NPZ chronological split and merge.
- Works for IMU-only or BOTH IMU routes (acc/gyr). (auto-detectable)
- Produces train_all.npz / val_all.npz / test_all.npz and a manifest.json.

Usage example (IMU only):
  python tools/split_eachseq_merge_npz.py \
    --root F:/SLAMdata/_cache/imu_eachseq \
    --out_dir F:/SLAMdata/_cache/imu_split \
    --pattern "*.npz" --mode chronological --split 0.70 0.15 0.15 --route imu

"""

import argparse, glob, json, os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import re

IMU_KEYS = ["TS_IMU","X_IMU_ACC","X_IMU_GYR","E2_IMU_ACC","E2_IMU_GYR","MASK_IMU","SEQ_NAME"]

ALIASES = {
    # IMU aliases (keep strict)
    "TS_IMU":      ["TS_IMU"],
    "X_IMU_ACC":   ["X_IMU_ACC"],
    "X_IMU_GYR":   ["X_IMU_GYR"],
    "E2_IMU_ACC":  ["E2_IMU_ACC"],
    "E2_IMU_GYR":  ["E2_IMU_GYR"],
    "MASK_IMU":    ["MASK_IMU"],
    "SEQ_NAME":    ["SEQ_NAME","seq_name","seq"],
}

def find_key(d: np.lib.npyio.NpzFile, canonical: str):
    for k in ALIASES.get(canonical, [canonical]):
        if k in d.files:
            return k
    return None

def detect_route(d: np.lib.npyio.NpzFile) -> str:
    has_imu = (find_key(d, "X_IMU_ACC") is not None) or (find_key(d, "E2_IMU_ACC") is not None)
    if has_imu: return "imu"
    return "none"

def get_len_from_any(arrs: List[np.ndarray]) -> int:
    for a in arrs:
        if a is None: continue
        # accept 1D (T,) or 2D (T,D) or 3D (T,...) -> take first dim as T
        if a.ndim >= 1:
            return int(a.shape[0])
    return 0

def slice_by_mode(T: int, mode: str, split: Tuple[float,float,float]) -> Tuple[slice,slice,slice]:
    p_train, p_val, p_test = split
    n_tr = int(np.floor(T * p_train))
    n_va = int(np.floor(T * p_val))
    n_te = T - n_tr - n_va
    if mode == "chronological":
        i_tr = slice(0, n_tr)
        i_va = slice(n_tr, n_tr+n_va)
        i_te = slice(n_tr+n_va, T)
    else:
        # for reproducibility, random split should be done outside with perm index;
        # here we keep chronological for stability
        i_tr = slice(0, n_tr)
        i_va = slice(n_tr, n_tr+n_va)
        i_te = slice(n_tr+n_va, T)
    return i_tr, i_va, i_te

def ensure_2d(a: np.ndarray) -> np.ndarray:
    if a is None: return None
    if a.ndim == 1:
        return a.copy()
    return a


def base_seq_name(stem: str) -> str:
    """Strip trailing _T<win>_S<stride> to get sequence base name."""
    m = re.match(r'^(.*)_T\d+_S\d+$', stem)
    return m.group(1) if m else stem


def normalize_imu_dict(d: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    # Optional: keep as-is; only include keys that exist
    out = {}
    for k in IMU_KEYS:
        kk = find_key(d, k)
        if kk is not None:
            out[k] = d[kk]
    # determine T from any IMU time series
    T = get_len_from_any([out.get("TS_IMU"), out.get("X_IMU_ACC"), out.get("X_IMU_GYR")])
    if T == 0:
        return {}
    # basic sanity
    for k in list(out.keys()):
        if out[k].shape[0] != T:
            raise ValueError(f"[shape-mismatch] {k} shape {out[k].shape}, T={T}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir of per-seq npz")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--mode", choices=["chronological","random"], default="chronological")
    ap.add_argument("--split", nargs=3, type=float, default=[0.70,0.15,0.15])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", choices=["auto","imu"], default="auto",
                    help="which route keys are required. 'auto' detects per file.")
    ap.add_argument("--test_seqs", type=str, default="",
                    help="Comma-separated base seq names to assign wholly to test (e.g., 'V1_03_difficult,V2_02_medium')")
    ap.add_argument("--val_seqs", type=str, default="",
                    help="Comma-separated base seq names to assign wholly to val")
    ap.add_argument("--exclude_seqs", type=str, default="")
    args = ap.parse_args()
    
    # Ensure VIS route is not used
    if args.route not in ("auto", "imu"):
        raise ValueError(f"Only 'auto' and 'imu' routes are supported. Got: {args.route}")

    os.makedirs(args.out_dir, exist_ok=True)

    def _parse_seq_list(s: str):
        return set(base_seq_name(x.strip()) for x in s.split(",") if x.strip())

    test_set = _parse_seq_list(args.test_seqs)
    val_set  = _parse_seq_list(args.val_seqs) - test_set
    exclude_set = _parse_seq_list(args.exclude_seqs)
    paths = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not paths:
        print(f"[warn] no files matched: {args.root}/{args.pattern}")

    merged = {"train": {}, "val": {}, "test": {}}
    manifest = {"splits": {"train": [], "val": [], "test": []}}

    for p in paths:
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f"[skip] {Path(p).name} cannot open: {e}")
            continue

        want = args.route
        if want == "auto":
            want = detect_route(d)  # imu / none

        imu_dict = normalize_imu_dict(d) if want == "imu" else {}

        if want == "imu" and not imu_dict:
            print(f"[skip] {Path(p).name} missing IMU keys.")
            continue
        if want == "none":
            print(f"[skip] {Path(p).name} none route detected.")
            continue

        # decide T based on available dict
        T = get_len_from_any([imu_dict.get("TS_IMU"), imu_dict.get("X_IMU_ACC"), imu_dict.get("X_IMU_GYR")]) if imu_dict else 0
        if T <= 1:
            print(f"[skip] {Path(p).name} T={T} too short.")
            continue

        # Derive SEQ_NAME from file stem (e.g., V1_01_easy_T512_S256)
        seq_name = Path(p).stem
        seq_arr = np.array([seq_name]*T)
        base = base_seq_name(seq_name)
        if base in exclude_set:
            continue

        def add_split(name: str, sub: Dict[str,np.ndarray], sl: slice, seq_arr: np.ndarray):
            if not sub:
                return
            bucket = merged[name]
            added_any = False
            for k, v in sub.items():
                vv = v[sl]
                if getattr(vv, 'shape', None) is not None and vv.shape[0] == 0:
                    continue
                if k not in bucket:
                    bucket[k] = [vv]
                else:
                    bucket[k].append(vv)
                added_any = True
            # Ensure SEQ_NAME is present only if we actually added something
            if added_any and ('SEQ_NAME' not in sub):
                sn = seq_arr[sl]
                if getattr(sn, 'shape', None) is not None and sn.shape[0] == 0:
                    return
                if 'SEQ_NAME' not in bucket:
                    bucket['SEQ_NAME'] = [sn]
                else:
                    bucket['SEQ_NAME'].append(sn)

        if test_set or val_set:
            # Assign whole sequence to the specified split
            if base in test_set:
                add_split("test", imu_dict, slice(0, T), seq_arr)
                manifest["splits"]["test"].append({"file": Path(p).name, "idx": [0, T]})
            elif base in val_set:
                add_split("val", imu_dict, slice(0, T), seq_arr)
                manifest["splits"]["val"].append({"file": Path(p).name, "idx": [0, T]})
            else:
                add_split("train", imu_dict, slice(0, T), seq_arr)
                manifest["splits"]["train"].append({"file": Path(p).name, "idx": [0, T]})
        else:
            # Fallback: split within sequence by time
            i_tr, i_va, i_te = slice_by_mode(T, args.mode, tuple(args.split))
            add_split("train", imu_dict, i_tr, seq_arr)
            add_split("val",   imu_dict, i_va, seq_arr)
            add_split("test",  imu_dict, i_te, seq_arr)

            manifest["splits"]["train"].append({"file": Path(p).name, "idx": [i_tr.start, i_tr.stop]})
            manifest["splits"]["val"].append({"file": Path(p).name, "idx": [i_va.start, i_va.stop]})
            manifest["splits"]["test"].append({"file": Path(p).name, "idx": [i_te.start, i_te.stop]})

    # save
    for split in ["train","val","test"]:
        if not merged[split]:
            print(f"[warn] {split}: nothing to merge")
            continue
        out = {}
        for k, arrs in merged[split].items():
            try:
                out[k] = np.concatenate(arrs, axis=0)
            except Exception:
                # fallback to list-save if shapes differ slightly
                out[k] = np.array(arrs, dtype=object)
        np.savez(os.path.join(args.out_dir, f"{split}_all.npz"), **out)
        print(f"[ok] wrote {split}_all.npz with keys: {list(out.keys())}")

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[ok] wrote manifest.json")
    print("[done]")
    
if __name__ == "__main__":
    main()
