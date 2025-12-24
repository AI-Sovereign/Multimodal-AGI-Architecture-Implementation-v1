import torch
import torch.nn as nn
import torch.nn.functional as F

class OjaPlasticLayer(nn.Module):
    """
    Synaptic Plasticity: Δw = η * (S * C) * (yxᵀ - y²w)
    Ensures unit-norm stability during real-time learning.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) * 0.01)

    def forward(self, x):
        return F.linear(x, self.weight)

    def apply_plasticity(self, x, y, eta, surprisal, causal_c):
        with torch.no_grad():
            modulator = surprisal * causal_c
            hebbian = torch.mm(y.t(), x) / x.size(0)
            forgetting = (torch.sum(y**2, dim=0).view(-1, 1)) * self.weight
            delta_w = eta * modulator * (hebbian - forgetting)
            self.weight.add_(delta_w)
            self.weight.div_(torch.norm(self.weight, dim=1, keepdim=True) + 1e-8)

class HTSP_Unit(nn.Module):
    """Dual-Clock Recurrence: Fast (Reaction) & Slow (Deliberation)."""
    def __init__(self, dim):
        super().__init__()
        self.fast = nn.GRU(dim, dim, batch_first=True)
        self.slow = nn.LSTM(dim, dim, batch_first=True)
        self.fusion = nn.Linear(dim * 2, dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        f_o, _ = self.fast(x)
        s_o, _ = self.slow(x)
        combined = torch.cat([f_o[:, -1, :], s_o[:, -1, :]], dim=-1)
        return self.fusion(combined)

class SovereignUltima(nn.Module):
    def __init__(self):
        super().__init__()
        # Jameen Machine: 21 Discrete Encoders
        self.encoders = nn.ModuleList([nn.Linear(64, 128) for _ in range(21)])
        
        # Dual-Stream Fusion
        self.ventral_net = nn.Sequential(nn.Linear(128 * 10, 512), nn.ReLU())
        self.dorsal_htsp = HTSP_Unit(128 * 11)
        
        # Global Workspace Theory (GWT) Bottleneck
        self.gwt_bottleneck = OjaPlasticLayer(512 + 1408, 256)
        
        # Basal Ganglia Executive Gating
        self.bg_direct = nn.Linear(256, 256) # Go
        self.bg_indirect = nn.Linear(256, 256) # No-Go
        
        self.action_head = nn.Linear(256, 12)
        self.vigilance_head = nn.Linear(256, 1)

    def forward(self, x_tensor, memory_context):
        # 1. Discrete Encoding
        latents = [self.encoders[i](x_tensor[:, i, :]) for i in range(21)]
        v_stream = torch.cat(latents[:10], dim=-1)
        d_stream = torch.cat(latents[10:], dim=-1)
        
        # 2. Stream Processing
        v_out = self.ventral_net(v_stream)
        d_out = self.dorsal_htsp(d_stream.unsqueeze(1))
        
        # 3. Global Workspace Fusion
        combined = torch.cat([v_out, d_out], dim=-1)
        workspace = self.gwt_bottleneck(combined)
        
        # 4. Basal Ganglia Gating
        go, no_go = torch.sigmoid(self.bg_direct(workspace)), torch.sigmoid(self.bg_indirect(workspace))
        gated = workspace * (go - no_go)
        
        return self.action_head(gated), gated, combined, torch.sigmoid(self.vigilance_head(gated))
