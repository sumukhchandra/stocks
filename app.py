"""
Root entry point for Streamlit application (Hugging Face / Local / Cloud).
Delegates execution to frontend/app.py.
"""
import os
import sys
import runpy

# Ensure root directory is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Execute frontend/app.py as main script
app_path = os.path.join(ROOT_DIR, "frontend", "app.py")
runpy.run_path(app_path, run_name="__main__")
