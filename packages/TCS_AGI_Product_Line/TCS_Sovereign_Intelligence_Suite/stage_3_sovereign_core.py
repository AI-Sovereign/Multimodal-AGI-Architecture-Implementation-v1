import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
import numpy as np

class GPlasticOjaLayer(nn.Module):
    def __init__(self, in_f, out_f, eta=0.001):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(eta), requires_grad=False)
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)

    def forward(self, x, surprisal=1.0, causal_signal=1.0):
        y = F.linear(x, self.weight)
        if self.training:
            with torch.no_grad():
                # Neuromodulatory Plasticity: Δw = η * (S * C) * (yx - y²w)
                mod_eta = self.eta * surprisal * causal_signal
                dw = mod_eta * (torch.mm(y.t(), x) - torch.mm(y.t(), y) @ self.weight)
                self.weight.add_(dw)
        return torch.tanh(y)

class SovereignOmega(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Basal Ganglia Dual-Pathway Gating (Go/No-Go)
        self.gate_go = nn.Linear(1024, 512)
        self.gate_nogo = nn.Linear(1024, 512)
        
        # 2. Dual-Stream Fusion (Ventral/Dorsal)
        self.ventral_stream = nn.Linear(4096 + 1024, 1024) # What (Visual + Ling)
        self.dorsal_stream = nn.Linear(512 + 768, 1024)    # How (Somato + Telem)
        
        # 3. Global Workspace (GWT) & Expert Networks
        self.workspace_fusion = nn.Linear(2048 + 1024, 2048) # Includes Memory Context
        self.experts = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(8)])
        
        # 4. Axiomatic Knowledge Layer (AKL)
        self.axiom_vault = nn.Parameter(torch.randn(64, 2048))
        self.axiom_gate = nn.Linear(2048, 64)
        
        # 5. Plasticity & Vigilance (LIF Head)
        self.plasticity = GPlasticOjaLayer(2048, 1024)
        self.vigilance_head = nn.Linear(1024, 1)
        self.surprisal_head = nn.Linear(2048, 1)
        
        # Causal Adjacency Mask
        self.register_buffer("causal_mask", torch.from_numpy(np.load("causal_mask.npy")).float())

    def forward(self, vis, somato, lang, telem, mem_context):
        # Basal Ganglia Inhibitory Logic
        go = torch.sigmoid(self.gate_go(lang))
        nogo = torch.sigmoid(self.gate_nogo(lang))
        gated_lang = lang * (go - nogo)
        
        # Dual-Stream Processing
        v_feat = torch.relu(self.ventral_stream(torch.cat([vis, gated_lang], dim=1)))
        d_feat = torch.relu(self.dorsal_stream(torch.cat([somato, telem], dim=1)))
        
        # GWT Synthesis with Recursive Hindsight
        combined = torch.cat([v_feat, d_feat, mem_context], dim=1)
        workspace = torch.relu(self.workspace_fusion(combined))
        
        # Axiomatic Injection
        axioms = torch.mm(F.softmax(self.axiom_gate(workspace), dim=-1), self.axiom_vault)
        workspace = (workspace + axioms) @ self.causal_mask
        
        # Multi-Expert Executive Control
        expert_outputs = torch.stack([exp(workspace) for exp in self.experts], dim=1)
        workspace = torch.mean(expert_outputs, dim=1)
        
        # Neuromodulation & Plasticity
        surprisal = torch.sigmoid(self.surprisal_head(workspace))
        latent = self.plasticity(workspace, surprisal.mean())
        
        # LIF Spiking Vigilance
        vigilance = torch.sigmoid(self.vigilance_head(latent))
        action = torch.tanh(nn.Linear(1024, 512).to(latent.device)(latent))
        
        return action * (vigilance > 0.3).float(), vigilance

if __name__ == "__main__":
    model = SovereignOmega()
    logger.success("MODELING COMPLETE: Sovereign Core wired.")
    torch.save(model.state_dict(), "sovereign.pth")
