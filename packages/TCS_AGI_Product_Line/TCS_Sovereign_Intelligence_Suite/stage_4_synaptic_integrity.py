import torch
import numpy as np
from loguru import logger

def evaluate_synaptic_health():
    logger.info("PHASE 4: Commencing Synaptic Stress Test...")
    weights = torch.load("sovereign.pth")
    fusion_w = weights['workspace_fusion.weight'].numpy()
    
    # Synaptic Norm Stability (Oja Compliance)
    norm = np.linalg.norm(fusion_w, axis=1).mean()
    logger.info(f"Synaptic Norm Stability: {norm:.4f}")
    
    # Acyclicity Validation (NOTEARS)
    mask = np.load("causal_mask.npy")
    h_val = np.trace(np.exp(mask @ mask)) - mask.shape[0]
    logger.info(f"NOTEARS Acyclicity Score: {h_val:.4f}")
    
    if norm < 5.0 and h_val < 1e-5:
        logger.success("EVALUATION PASSED: Sovereign Integrity Confirmed.")
    else:
        logger.warning("EVALUATION WARN: Potential Structural Drift.")

if __name__ == "__main__":
    evaluate_synaptic_health()
