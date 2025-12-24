from fastapi import FastAPI
import uvicorn
import torch
import numpy as np
from avalon_03_core import SovereignUltima
from avalon_04_evaluation import RecursiveHindsight

app = FastAPI(title="AVALON V29-ULTIMA")
model = SovereignUltima()
memory = RecursiveHindsight()
causal_signal = 0.95

@app.post("/telemetry")
async def process(payload: dict):
    # Convert input to Jameen Tensor (1, 21, 64)
    raw_data = np.array(payload['data']).reshape(1, 21, 64)
    x = torch.tensor(raw_data).float()
    
    # Hindsight retrieval
    h_context = memory.retrieve(torch.randn(1, 256))
    
    # Inference
    action, latent, combined, vigilance = model(x, h_context)
    
    # Real-time Oja Update (Neuromodulation)
    surprisal = torch.tensor([0.5]) # Mock surprisal
    model.gwt_bottleneck.apply_plasticity(combined, latent, 0.001, surprisal, causal_signal)
    
    # Store in ERM
    memory.store(latent)
    
    return {
        "status": "SOVEREIGN_NOMINAL",
        "action": action.tolist()[0],
        "vigilance": float(vigilance)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
