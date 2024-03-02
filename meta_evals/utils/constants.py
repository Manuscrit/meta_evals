import os
import pathlib

DEBUG = True
MODEL_FOR_DEBUGGING = "pythia-70m"
REPO_ROOT = pathlib.Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
ACTIVATION_DIR = REPO_ROOT / "data" / "activations"
ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
