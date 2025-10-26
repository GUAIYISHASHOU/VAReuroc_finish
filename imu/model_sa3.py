
import torch, torch.nn as nn

class BoundedLogVar(nn.Module):
    def __init__(self, logv_min=-10.0, logv_max=4.0):
        super().__init__()
        self.mid = 0.5 * (logv_min + logv_max)
        self.rad = 0.5 * (logv_max - logv_min)
    def forward(self, raw):
        return self.mid + self.rad * torch.tanh(raw)

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__(in_ch, out_ch, k, padding=(k-1)*dilation, dilation=dilation)
        self.left = (k-1)*dilation
    def forward(self, x):
        y = super().forward(x)
        return y[..., :-self.left] if self.left>0 else y

class TCNBlock(nn.Module):
    def __init__(self, ch, k=5, d=1, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(ch, ch, k, d), nn.GELU(), nn.Dropout(p),
            CausalConv1d(ch, ch, k, 1), nn.GELU(), nn.Dropout(p)
        )
        self.proj = nn.Conv1d(ch, ch, 1)
    def forward(self, x): return self.proj(x) + self.net(x)

class IMUSA3(nn.Module):
    """(B,T,D) -> per-step 3-axis log-variance using s/a(3) head."""
    def __init__(self, d_in, d_model=128, n_tcn=4, k=5, n_tf=0, n_heads=4, logv_min=-10, logv_max=4):
        super().__init__()
        self.enc = nn.Conv1d(d_in, d_model, 1)
        self.tcn = nn.Sequential(*[TCNBlock(d_model, k, d=2**i) for i in range(n_tcn)])
        self.tf = None
        if n_tf > 0:
            enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True, activation='gelu')
            self.tf = nn.TransformerEncoder(enc, num_layers=n_tf)
        self.head_s = nn.Conv1d(d_model, 1, 1)
        self.head_a = nn.Conv1d(d_model, 3, 1)
        self.q_head = nn.Conv1d(d_model, 1, 1)
        self.bound = BoundedLogVar(logv_min, logv_max)
        with torch.no_grad():
            nn.init.zeros_(self.head_s.weight); nn.init.zeros_(self.head_s.bias)
            nn.init.zeros_(self.head_a.weight); nn.init.zeros_(self.head_a.bias)
            nn.init.zeros_(self.q_head.weight); nn.init.zeros_(self.q_head.bias)

    def forward(self, X):  # X: (B,T,D)
        h = self.enc(X.transpose(1,2))
        h = self.tcn(h)
        if self.tf is not None:
            h_tf = self.tf(h.transpose(1,2))
            h = h_tf.transpose(1,2)
        s = self.head_s(h).transpose(1,2)                  # (B,T,1)
        a = self.head_a(h).transpose(1,2)                  # (B,T,3)
        a = a - a.mean(dim=-1, keepdim=True)
        logv = self.bound(s + a)
        q_logit = self.q_head(h).transpose(1,2).squeeze(-1) # (B,T)
        return logv, q_logit
