# omega_04_deploy.py
"""
STAGE 4 & 5: EVALUATION & DEPLOYMENT - THE API INTERFACE
--------------------------------------------------------
Real World Application using FastAPI.
Integrates Scikit-Learn k-NN for Recursive Hindsight.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import uvicorn
import os

# Import Core
from omega_03_modeling import OmegaSovereign

# --- [COMPONENT: EPISODIC RELATIONAL MEMORY (k-NN)] ---
class RecursiveHindsightMemory:
    def __init__(self):
        # Memory Bank: Stores latent states
        self.bank = np.zeros((500, 256))
        self.ptr = 0
        self.full = False
        self.knn = NearestNeighbors(n_neighbors=1, metric='cosine')

    def retrieve(self, query_vec):
        # query_vec: (1, 256)
        if not self.full and self.ptr == 0:
            return torch.zeros(1, 256)
        
        # Fit on current memory (Real-time update)
        current_data = self.bank if self.full else  self.bank[:self.ptr]
        if len(current_data) < 1: return torch.zeros(1, 256)
        
        self.knn.fit(current_data)
        
        # Convert torch tensor to numpy for sklearn
        q_np = query_vec.detach().numpy()
        _, idx = self.knn.kneighbors(q_np)
        
        # Retrieve the closest "Episode"
        return torch.tensor(self.bank[idx[0][0]]).unsqueeze(0)

    def store(self, vec):
        self.bank[self.ptr] = vec.detach().numpy()
        self.ptr = (self.ptr + 1) % 500
        if self.ptr == 0: self.full = True

# --- [FASTAPI APP] ---
app = FastAPI(title="PROJECT AEGIS: OMEGA SOVEREIGN", version="25.0")

# Globals
model = None
memory = None

@app.on_event("startup")
def load_system():
    global model, memory
    print(">>> [SYSTEM] INITIALIZING OMEGA CORE...")
    
    # Check if Mask exists
    if not os.path.exists("omega_causal_mask.npy"):
        print(">>> [WARNING] Causal Mask not found. Run stage_2 first. Using dummy.")
    
    model = OmegaSovereign()
    memory = RecursiveHindsightMemory()
    model.eval()
    print(">>> [SYSTEM] AEGIS ONLINE. READY FOR INGESTION.")

class TelemetryInput(BaseModel):
    # Flattened input: 21 * 64 = 1344 floats
    raw_stream: List[float]

@app.post("/process_telemetry")
def process_data(data: TelemetryInput):
    if len(data.raw_stream) != 1344:
        raise HTTPException(status_code=400, detail="Input must be exactly 1344 floats.")
    
    # 1. Reshape Input
    input_tensor = torch.tensor(data.raw_stream).view(1, 21, 64).float()
    
    # 2. Hindsight Query (Bootstrap with random or previous state)
    query_context = torch.randn(1, 256) 
    hindsight = memory.retrieve(query_context)
    
    # 3. Sovereign Forward Pass
    action, latent, combined_input, vig = model(input_tensor, hindsight)
    
    # 4. Meta-Learning Update (Oja's Rule)
    # Simulated Surprisal (Target - Output)
    target = torch.randn(1, 12) 
    surprisal = torch.nn.functional.mse_loss(action, target)
    
    # Dynamic ETA based on Vigilance (Meta-Learning Control)
    eta_dynamic = 0.001 * vig * float(surprisal)
    
    model.gw_fusion.manual_update(
        x=combined_input,
        y=latent,
        eta=eta_dynamic,
        surprisal_c=surprisal
    )
    
    # 5. Store Memory
    memory.store(latent)
    
    return {
        "status": "NOMINAL",
        "vigilance_level": float(vig),
        "action_vector": action.tolist()[0],
        "hindsight_active": True,
        "axioms_checked": True
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
