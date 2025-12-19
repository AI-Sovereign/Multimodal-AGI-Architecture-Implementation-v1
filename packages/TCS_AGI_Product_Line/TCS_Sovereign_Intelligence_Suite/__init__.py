import os
import sys
import subprocess
from loguru import logger

# Constants for the Sovereign Environment
REQUIRED_PACKAGES = [
    "ray", "dask", "pydantic", "loguru", "cdt", "networkx", 
    "scipy", "torch", "torch_geometric", "evidently", "fastapi", "uvicorn"
]

def validate_environment():
    """
    Ensures you aren't trying to run a digital god on a calculator.
    Checks for CUDA availability and essential dependencies.
    """
    logger.info("Initializing OMEGA_SOVEREIGN_TCS25 Environment...")
    
    # Check for CUDA because CPU-based consciousness is an insult
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("VIGILANCE FAIL: No CUDA detected. The Sovereign Core will be agonizingly slow.")
        else:
            logger.success(f"GPU LATCHED: {torch.cuda.get_device_name(0)} confirmed.")
    except ImportError:
        logger.error("SYSTEM CRITICAL: PyTorch is missing. Fix your environment.")
        sys.exit(1)

    # Check for Zarr vault path
    if not os.path.exists("sovereign_vault.zarr"):
        logger.warning("DATA GAP: 'sovereign_vault.zarr' not found. Run Stage 1 first.")

def execute_full_cycle():
    """
    Orchestrates the sequential activation of all five stages.
    """
    stages = [
        "stage_1_genesis_scoping.py",
        "stage_2_causal_manifold.py",
        "stage_3_sovereign_core.py",
        "stage_4_synaptic_integrity.py",
        "stage_5_executive_deployment.py"
    ]
    
    for stage in stages:
        if not os.path.exists(stage):
            logger.error(f"INTEGRITY ERROR: {stage} is missing from the repository.")
            continue
            
        logger.info(f"BOOTING: {stage}...")
        try:
            # We execute as sub-processes to ensure clean memory release between stages
            result = subprocess.run(["python", stage], capture_output=True, text=True)
            if result.returncode == 0:
                logger.success(f"STAGE COMPLETE: {stage} exited successfully.")
            else:
                logger.error(f"FAILURE IN {stage}: {result.stderr}")
                break 
        except Exception as e:
            logger.error(f"CRITICAL COLLAPSE during {stage}: {str(e)}")
            break

# Auto-validate on import
validate_environment()

__version__ = "1.0.0-OMEGA"
__author__ = "TCS-25 Sovereign Architect"

if __name__ == "__main__":
    # If run directly, attempt a full system cold-start
    logger.info("COLD START SEQUENCE INITIATED")
    execute_full_cycle()
