
import numpy as np
import torch
from torch.utils.data import Dataset

class IMUFrames(Dataset):
    """
    Reuses IMU NPZ format. Route: 'acc' or 'gyr'.
    Returns per-step e (not e^2), mask, and standardized features X.
    """
    def __init__(self, npz_path: str, route: str, scaler_npz: str | None = None, subset_ids: np.ndarray | None = None):
        d = np.load(npz_path, allow_pickle=True)
        self.route = route
        def pick(keys):
            for k in keys:
                if k in d.files:
                    return d[k]
            return None

        if route == "acc":
            X = pick(["X_IMU_ACC", "X_acc", "X"])
            E2 = pick(["E2_IMU_ACC", "Y_IMU_ACC"])
        else:
            X = pick(["X_IMU_GYR", "X_gyr", "X"])
            E2 = pick(["E2_IMU_GYR", "Y_IMU_GYR"])
        if X is None or E2 is None:
            raise ValueError(f"NPZ missing required keys for route={route!r}.")
        if E2.ndim == 2:  # (N,T) -> (N,T,1)->repeat 3
            E2 = E2[..., None]
        if E2.shape[-1] == 1:
            E2 = np.repeat(E2, 3, axis=-1)

        X = X.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.X = X
        self.E = np.sqrt(np.maximum(E2.astype(np.float32), 0.0))  # per-step |e|
        self.E = np.nan_to_num(self.E, nan=0.0, posinf=0.0, neginf=0.0)
        m = pick(["MASK_IMU", "mask"])
        self.M = m.astype(np.float32) if m is not None else np.ones(self.X.shape[:2], np.float32)
        self.seq = pick(["SEQ_NAME", "seq_name", "seq"])
        if self.seq is None:
            self.seq = np.array([f"seq_{i}" for i in range(len(self.X))])

        if subset_ids is not None:
            self.X = self.X[subset_ids]; self.E = self.E[subset_ids]; self.M = self.M[subset_ids]; self.seq = self.seq[subset_ids]

        # Fit/apply scaler (train-only fit if scaler_npz is None)
        if scaler_npz is None:
            mu, std = self._fit_scaler()
            self.scaler = {"mean": mu, "std": std}
        else:
            s = np.load(scaler_npz)
            self.scaler = {"mean": s["mean"], "std": s["std"]}
        self.X = (self.X - self.scaler["mean"]) / (self.scaler["std"] + 1e-8)

    def _fit_scaler(self):
        m = self.M > 0.5
        Xv = []
        for i in range(self.X.shape[0]):
            Xi = self.X[i][m[i]]
            if Xi.size:
                Xv.append(Xi)
        Xv = np.concatenate(Xv, axis=0) if len(Xv) else self.X.reshape(-1, self.X.shape[-1])
        mu = np.nanmean(Xv, 0).astype(np.float32)
        std = np.nanstd(Xv, 0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)
        return mu, std

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx):
        return {
            "X": torch.from_numpy(self.X[idx]),          # (T,D)
            "E": torch.from_numpy(self.E[idx]),          # (T,3)
            "M": torch.from_numpy(self.M[idx]),          # (T,)
        }
