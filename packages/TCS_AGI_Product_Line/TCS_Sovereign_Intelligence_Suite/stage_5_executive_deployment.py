from fastapi import FastAPI
import torch
import numpy as np
import pickle
from stage_3_sovereign_core import SovereignOmega
from sklearn.neighbors import NearestNeighbors

app = FastAPI(title="TCS-25 OMEGA EXECUTIVE")

# Global State Load
model = SovereignOmega()
model.load_state_dict(torch.load("sovereign.pth"))
model.eval()

with open("erm_graph.pkl", "rb") as f:
    ERM = pickle.load(f)
    FEATURES = np.array([ERM.nodes[i]['features'] for i in ERM.nodes])
    KNN = NearestNeighbors(n_neighbors=1).fit(FEATURES)

@app.post("/invoke")
async def invoke(vis: list, somato: list, lang: list, telem: list):
    # Recursive Hindsight Retrieval
    query = np.array(lang).reshape(1, -1)
    _, idx = KNN.kneighbors(query)
    mem_context = torch.tensor(FEATURES[idx[0]], dtype=torch.float32)
    
    # Tensor Conversion
    v_t = torch.tensor([vis], dtype=torch.float32)
    s_t = torch.tensor([somato], dtype=torch.float32)
    l_t = torch.tensor([lang], dtype=torch.float32)
    t_t = torch.tensor([telem], dtype=torch.float32)
    
    with torch.no_grad():
        action, vigilance = model(v_t, s_t, l_t, t_t, mem_context)
        
    return {
        "status": "SOVEREIGN_AUTHORIZED",
        "vigilance": float(vigilance.mean()),
        "action_vector": action.tolist()[0][:5]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
