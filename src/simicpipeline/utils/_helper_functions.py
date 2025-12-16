# ------------------------------------------------------------------------------
# Title: _heleper_functions.py
# Author: Irene
# Date: 24/01/2025
# Version: v1
# Description: Script with custom functions to be used in other scripts
# ------------------------------------------------------------------------------

import joblib
import pickle
from pathlib import Path
import subprocess
import sys
# Salve functions
def write_pickle(obj, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, protocol=4)
        print(f"Pickle method succeeded, saved to {file_path}")
    except OverflowError as e:
        print(f"Pickle failed due to OverflowError: {e}")
    except Exception as e:
        print(f"Pickle failed with error: {e}")

def write_joblib(obj, file_path):
    try:
        joblib.dump(obj, file_path)
        print(f"Joblib method succeeded, saved to {file_path}")
    except Exception as e:
        print(f"Joblib failed with error: {e}")

def read_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Pickle failed to load with error: {e}")


def read_joblib(file_path):
    try:
        with open(file_path, 'rb') as f:
            return joblib.load(f)
    except Exception as e:
        print(f"Joblib failed to load with error: {e}")

# Install packages

def install_package(package_name):
   """Install a package using pip3."""
   subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])