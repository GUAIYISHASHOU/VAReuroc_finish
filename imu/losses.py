
import torch

def nll_student_t_diag(logv_axes, e_axes, mask, nu=6.0):
    """Student-t NLL for diagonal 3-axis residuals."""
    nu = max(float(nu), 2.1)
    var = torch.exp(logv_axes)
    e2_over = (e_axes**2) / (nu * var + 1e-12)
    e2_over = torch.clamp(e2_over, max=1e6)
    nll = 0.5 * (logv_axes + (nu+1.0) * torch.log1p(e2_over))
    nll = torch.nan_to_num(nll, posinf=50.0, neginf=50.0)
    nll = nll.sum(dim=-1)  # sum over axes
    if mask.ndim == 3: mask = mask[...,0]
    finite_e = torch.isfinite(e_axes).all(dim=-1)
    w = (mask > 0.5) & finite_e
    nll = torch.nan_to_num(nll, posinf=50.0, neginf=50.0)
    return (nll[w].mean() if w.any() else nll.mean())
