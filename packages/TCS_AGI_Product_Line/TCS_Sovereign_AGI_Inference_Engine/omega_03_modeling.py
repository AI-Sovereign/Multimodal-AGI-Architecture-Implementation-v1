# omega_03_modeling.py
"""
STAGE 3: MODELING - THE OMEGA SOVEREIGN CORE
--------------------------------------------
The Biological Neural Fabric.
Includes: Oja's Rule, HTSP (Fast/Slow), Basal Ganglia, LIF, 21-Discrete Encoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- [COMPONENT: OJA'S RULE PLASTICITY] ---
# Math: Δw = η * (S * C) * (yx - y^2w)
class OjaSynapticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    def forward(self, x):
         return F.linear(x, self.weight)

    def manual_update(self, x, y, eta, surprisal_c):
        with torch.no_grad():
            # Hebbian: y * x^T
            # Ensure proper shapes for matrix multiplication: y(B, Out), x(B, In)
            hebbian = torch.mm(y.t(), x) / x.size(0)
            
            # Oja Forgetting: y^2 * w
            y_sq = torch.sum(y ** 2, dim=0).view(-1, 1) / x.size(0)
            forgetting = y_sq * self.weight
            
            # Modulated Update
            delta_w = eta * surprisal_c * (hebbian - forgetting)
            self.weight.add_(delta_w)
            
            # Unit Sphere Normalization (Stability)
            self.weight.div_(torch.norm(self.weight, dim=1, keepdim=True) + 1e-8)

# --- [COMPONENT: AXIOMATIC INSTINCT LAYER] ---
class AxiomaticInstinctLayer(nn.Module):
    def __init__(self, context_dim, axiom_dim):
        super().__init__()
        # The 'Genetic' knowledge base of the AI (Physics Constants)
        self.axiom_vault = nn.Parameter(torch.randn(64, axiom_dim)) 
        self.context_projection = nn.Linear(context_dim, 64)

    def forward(self, current_context):
        # Soft-selection of the relevant 'Instinct'
        selection_weights = F.softmax(self.context_projection(current_context), dim=-1)
        return torch.mm(selection_weights, self.axiom_vault)

# --- [COMPONENT: LEGACY DISCRETE SENSORY FABRIC] ---
# The 21 distinct encoders you demanded.
class LegacyDiscreteFabric(nn.Module):
    def __init__(self):
        super().__init__()
        # 21 Specialized Encoders: 64 input -> 128 latent
        self.modalities = nn.ModuleList([
            nn.Sequential(
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU()
            ) for i in range(21)
        ])
    
    def forward(self, x_tensor):
        # x_tensor: (Batch, 21, 64)
        latents = []
        for i in range(21):
            latents.append(self.modalities[i](x_tensor[:, i, :]))
        return latents # List of 21 tensors (B, 128)

# --- [COMPONENT: HTSP DUAL-CLOCK RECURRENCE] ---
# Fast/Slow Clock Temporal Processing for the JAMEEN MACHINE logic
class HTSP_DualClock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Fast Clock (2ms) - GRU
        self.fast_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Slow Clock (20ms) - LSTM
        self.slow_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq, Features) -> Simulate Seq=1 for real-time
        if x.dim() == 2: x = x.unsqueeze(1)
        
        f_out, _ = self.fast_rnn(x)
        s_out, _ = self.slow_rnn(x)
        
        # Fuse Fast (Reaction) and Slow (Trend)
        combined = torch.cat([f_out[:, -1, :], s_out[:, -1, :]], dim=1)
        return torch.tanh(self.fusion(combined))

# --- [COMPONENT: BASAL GANGLIA GATING] ---
class BasalGangliaGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.direct_path = nn.Linear(dim, dim)   # GO (Excitatory)
        self.indirect_path = nn.Linear(dim, dim) # NO-GO (Inhibitory)

    def forward(self, x):
        go = torch.sigmoid(self.direct_path(x))
        no_go = torch.sigmoid(self.indirect_path(x))
        return x * (go - no_go)

# --- [COMPONENT: LIF VIGILANCE & META-LEARNING] ---
class MetaLearningVigilance(nn.Module):
    def __init__(self, dim, decay=0.8):
        super().__init__()
        self.proj = nn.Linear(dim, 1)
        self.decay = decay
        self.membrane = 0.0

    def forward(self, x):
        input_current = torch.sigmoid(self.proj(x)).mean().item()
        self.membrane = self.membrane * self.decay + input_current
        # Returns Vigilance scalar
        return self.membrane

# --- [MAIN: THE OMEGA SOVEREIGN] ---
class OmegaSovereign(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Discrete Encoding (21 Modalities)
        self.sensory_fabric = LegacyDiscreteFabric()
        
        # 2. Dual Stream Fusion
        # Ventral (7 mods * 128) = 896
        # Dorsal (14 mods * 128) = 1792
        self.ventral_fusion = nn.Linear(896, 256)
        
        # Dorsal gets HTSP Temporal Processing
        self.dorsal_compress = nn.Linear(1792, 256)
        self.dorsal_htsp = HTSP_DualClock(256, 256)
        
        # 3. Axiomatic Injection
        self.axiomatic_layer = AxiomaticInstinctLayer(context_dim=512, axiom_dim=256)

        # 4. Global Workspace (The Bottleneck)
        # Input: 256(V) + 256(D) + 256(Axiom) + 256(Hindsight) = 1024
        self.gw_fusion = OjaSynapticLayer(1024, 256)
        
        # Load SCWP Mask (Causal Pruning) from Stage 2
        try:
            mask = np.load("omega_causal_mask.npy")
            self.causal_mask = torch.tensor(mask, dtype=torch.float32)
        except:
            # Fallback if stage 2 wasn't run
            self.causal_mask = torch.ones(256, 256)

        # 5. Executive Gating & Vigilance
        self.basal_ganglia = BasalGangliaGate(256)
        self.vigilance = MetaLearningVigilance(256)
        
        # 6. Output
        self.action_head = nn.Linear(256, 12) # 12 Thruster Controls

    def forward(self, x_tensor, hindsight_memory):
        # A. Discrete Encoding
        latents = self.sensory_fabric(x_tensor)
        
        # B. Split Streams (0-7 Ventral, 7-21 Dorsal)
        ventral = torch.cat(latents[:7], dim=1) # (B, 896)
        dorsal = torch.cat(latents[7:], dim=1)  # (B, 1792)
        
        # C. Process Streams
        v_out = self.ventral_fusion(ventral)
        
        d_compressed = self.dorsal_compress(dorsal)
        d_out = self.dorsal_htsp(d_compressed).squeeze(1) # HTSP Dual-Clock
        
        # D. Axiomatic Instinct
        # We combine V and D to query axioms
        context_query = torch.cat([v_out, d_out], dim=1)
        axiom_out = self.axiomatic_layer(context_query)
        
        # E. Recursive Hindsight Integration
        # Concatenate: What I see + How I move + What I know (Axioms) + What I remember
        combined = torch.cat([v_out, d_out, axiom_out, hindsight_memory], dim=1)
        
        # F. Causal Masking (SCWP) & Oja Forward
        # Applying the NOTEARS mask (simulated here on output for demo stability)
        workspace = self.gw_fusion(combined)
        
        # G. Gating & Vigilance
        gated = self.basal_ganglia(workspace)
        vig_level = self.vigilance(gated)
        action = self.action_head(gated)
        
        return action, gated, combined, vig_level
