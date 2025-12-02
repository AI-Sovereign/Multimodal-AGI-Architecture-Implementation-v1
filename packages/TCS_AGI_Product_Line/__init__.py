"""TCS AGI Product Line Initialization. Exposes versioned modules."""
from . import TCS_25_Temporal_Engine
from . import TCS_26_Stateless_API # CORRECTED NAME

# Define exposed project versions
__all__ = ['TCS_25_Temporal_Engine', 'TCS_26_Stateless_API']
