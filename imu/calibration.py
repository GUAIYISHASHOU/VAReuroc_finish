
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class Affine1D:
    alpha: float = 1.0
    beta: float = 0.0
    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.alpha * x + self.beta
    @staticmethod
    def fit_deming(x, y, lam: float = 1.0):
        x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
        xmu, ymu = x.mean(), y.mean()
        Sxx = np.var(x, ddof=1); Syy = np.var(y, ddof=1); Sxy = np.cov(x, y, ddof=1)[0,1]
        b = (Syy - lam*Sxx + np.sqrt((Syy - lam*Sxx)**2 + 4*lam*Sxy*Sxy)) / (2*Sxy + 1e-12)
        a = ymu - b * xmu
        return Affine1D(alpha=float(b), beta=float(a))
    def to_dict(self):
        return asdict(self)
    @staticmethod
    def from_dict(d):
        return Affine1D(**d)

def fit_affine_per_axis(pred_logv_axes, target_loge2_axes, mode: str = "deming", lam: float = 1.0):
    calibs = []
    P = pred_logv_axes.reshape(-1, 3)
    T = target_loge2_axes.reshape(-1, 3)
    for j in range(3):
        x = P[:, j]; y = T[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        if mode == "deming":
            cal = Affine1D.fit_deming(x[m], y[m], lam=lam)
        else:
            X = np.c_[x[m], np.ones(m.sum())]
            beta = np.linalg.lstsq(X, y[m], rcond=None)[0]
            cal = Affine1D(alpha=float(beta[0]), beta=float(beta[1]))
        calibs.append(cal)
    return calibs
