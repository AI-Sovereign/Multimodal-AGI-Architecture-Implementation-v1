# __init__.py

# Import the core version string from the top-level version file
# This is required for clean package metadata
try:
    from ..version import __version__
except ImportError:
    __version__ = "0.0.0.unknown"

# Define the public API of the Quantum Graph Product Line
# When adding future projects (e.g., v2.0), include their core classes here.
__all__ = [
    'QuantumGraphModel',
    'QuantumGraphEngine',
    '__version__'
]

# Optional: Import key components here for direct access (e.g., from package.module import Class)
# from .model import QuantumGraphModel
# from .engine import QuantumGraphEngine 
