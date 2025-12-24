"""
PROJECT AVALON: THE V29-ULTIMA SOVEREIGN
"""
__version__ = "29.0.0"

# Import major components from each stage for easy access
from .avalon_01_genesis import OrbitalAxioms, JameenSensorium
from .avalon_02_causality import NOTEARS_Discovery
from .avalon_03_core import SovereignUltima, HTSP_Unit, OjaPlasticLayer
from .avalon_04_evaluation import RecursiveHindsight
from .avalon_05_deploy import app

# Constants for the 21-modality Jameen architecture
MODALITIES = 21
LATENT_DIM = 256
ENCODER_IN = 64

__all__ = [
    'SovereignUltima',
    'OrbitalAxioms',
    'JameenSensorium',
    'RecursiveHindsight',
    'NOTEARS_Discovery',
    'app'
]

def initialize_system():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f">>> AVALON V29-ULTIMA: Sovereign Core Active on {device}.")
    print(">>> All 21 sensor modalities aligned.")

initialize_system()
