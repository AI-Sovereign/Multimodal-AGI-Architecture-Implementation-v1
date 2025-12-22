"""
PROJECT AEGIS: OMEGA SOVEREIGN (TCS-25)
Namespace Initialization & Hardware Arbitration
"""

import torch
import logging
from rich.logging import RichHandler

# 1. FORCE SYSTEM LOGGING
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("AEGIS_CORE")
logger.info("Initializing OMEGA Sovereign Environment...")

# 2. HARDWARE ARBITRATION
# Explicitly locked to CUDA GPU Fabric. 
if not torch.cuda.is_available():
    logger.critical("AEGIS FATAL ERROR: GPU NOT DETECTED. SYSTEM TERMINATED.")
    raise SystemError("Project AEGIS requires a CUDA-enabled GPU to maintain 110% efficiency.")

DEVICE = torch.device("cuda")
torch.set_default_dtype(torch.float32)
torch.backends.cudnn.benchmark = True
logger.info(f"AEGIS ONLINE: Utilizing CUDA High-Performance GPU Fabric.")

# 3. GLOBAL ARCHITECTURAL CONSTANTS
MODALITIES = 21
LATENT_DIM = 256
INPUT_DIM_PER_MODALITY = 64
TOTAL_INPUT_DIM = MODALITIES * INPUT_DIM_PER_MODALITY

# 4. NAMESPACE EXPORTS
from .omega_01_scoping import OmegaSensorium, AxiomaticPhysics
from .omega_03_modeling import OmegaSovereign

__all__ = [
    "DEVICE",
    "MODALITIES",
    "LATENT_DIM",
    "TOTAL_INPUT_DIM",
    "OmegaSensorium",
    "AxiomaticPhysics",
    "OmegaSovereign"
]

logger.info("Namespace Locked. All 21 Modalities Synced to GPU.")
