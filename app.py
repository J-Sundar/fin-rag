"""
app.py

Root entry point for Hugging Face Spaces and standard deployment targets.
Invokes the Streamlit application defined in app/app.py.
"""

import sys
import runpy
from pathlib import Path

# Add project root to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the core Streamlit application
runpy.run_path(str(ROOT_DIR / "app" / "app.py"), run_name="__main__")

