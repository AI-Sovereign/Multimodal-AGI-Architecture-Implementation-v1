"""TCS-25 Temporal Engine Initialization. Exposes primary API components."""
from . import config_consts
from .core_architecture import TemporalCausalSynthesizer # UPDATED
from .execution_engine import TCS_GeneralIntelligence # UPDATED

__all__ = [
    'config_consts', 
    'TemporalCausalSynthesizer', 
    'TCS_GeneralIntelligence'
]
