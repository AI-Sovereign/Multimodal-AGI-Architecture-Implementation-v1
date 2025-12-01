"""Multimodal AGI Architecture Implementation Top-Level Framework Initialization."""
import os
import glob
import importlib

# --- Dynamic Package Discovery ---
# 1. Define the directory to scan (current directory, which is 'packages/')
PACKAGE_DIR = os.path.dirname(__file__)
# 2. Find all directories that look like product lines (e.g., exclude __pycache__, etc.)
#    The pattern '*/' finds all directories.
discovered_modules = [
    os.path.basename(d) 
    for d in glob.glob(os.path.join(PACKAGE_DIR, '*')) 
    if os.path.isdir(d) and not os.path.basename(d).startswith('_')
]

# 3. Dynamically import and add to globals and __all__
__all__ = []
for module_name in discovered_modules:
    # Use importlib to perform the import and add it to the current namespace
    try:
        module = importlib.import_module(f'.{module_name}', package=__name__)
        globals()[module_name] = module
        __all__.append(module_name)
    except ImportError as e:
        # Professional logging: Alert if a folder can't be imported (e.g., if it's broken)
        print(f"Warning: Could not import product line '{module_name}'. Error: {e}")
# ---------------------------------

# Set a mandatory environment variable for the framework
os.environ["PROJECT_NAME"] = "TCS_AGI_FRAMEWORK"

# Note: __all__ is automatically populated by the loop above.
