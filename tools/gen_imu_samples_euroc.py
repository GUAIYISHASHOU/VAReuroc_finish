
#!/usr/bin/env python3
import numpy as np, argparse, os

def read_imu_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    t = data[:,0].astype(np.int64)
    gyr = data[:,1:4]; acc = data[:,4:7]
    return t, gyr, acc

def windowize(X, T, stride):
    N = (len(X) - T) // stride + 1
    idx = np.arange(0, N*stride, stride)[:,None] + np.arange(T)[None,:]
    return X[idx]

def robust_detrend(x, k=101):
    k = max(3, int(k)|1); pad = k//2
    xp = np.pad(x, ((0,0),(pad,pad),(0,0)), mode="edge")
    trend = np.empty_like(x)
    for i in range(x.shape[1]):
        w = xp[:,i:i+k]; trend[:,i] = np.median(w, axis=1)
    return x - trend

def compute_features(sig):
    diff = np.diff(sig, axis=1, prepend=sig[:,0:1])
    mag  = np.linalg.norm(sig, axis=-1, keepdims=True)
    return np.concatenate([sig, mag, diff], axis=-1)

def compute_residuals(sig):
    return robust_detrend(sig, k=101)

ap = argparse.ArgumentParser()
ap.add_argument("--euroc_root", required=True)
ap.add_argument("--seq", required=True)
ap.add_argument("--route", choices=["acc","gyr"], required=True)
ap.add_argument("--T", type=int, default=100)
ap.add_argument("--stride", type=int, default=50)
ap.add_argument("--out_npz", required=True)
args = ap.parse_args()

imu_csv = os.path.join(args.euroc_root, args.seq, "mav0", "imu0", "data.csv")
t, gyr, acc = read_imu_csv(imu_csv)
sig = acc if args.route=="acc" else gyr
W = windowize(sig, args.T, args.stride)
X = compute_features(W)
E = compute_residuals(W); E2 = E**2
MASK = np.ones((X.shape[0], X.shape[1]), np.float32)
SEQ = np.array([args.seq]*X.shape[0])

os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
np.savez_compressed(args.out_npz,
    **({ "X_IMU_ACC": X } if args.route=="acc" else { "X_IMU_GYR": X }),
    **({ "E2_IMU_ACC": E2 } if args.route=="acc" else { "E2_IMU_GYR": E2 }),
    MASK_IMU=MASK, SEQ_NAME=SEQ,
    META={"T":args.T,"stride":args.stride,"route":args.route}
)
print("Saved", args.out_npz, X.shape)
