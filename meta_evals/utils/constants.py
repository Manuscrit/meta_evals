import os
import pathlib

DEBUG = False
MODEL_FOR_DEBUGGING = "pythia-70m"
MODEL_FOR_REAL = (
    # "Llama-2-13b-chat-hf"
    "Mistral-7B-Instruct"
)
REPO_ROOT = pathlib.Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PLOT_DIR = REPO_ROOT / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVATION_DIR = REPO_ROOT / "data" / "activations"
ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)


def get_layer_filter():
    return "layer >= 1" if DEBUG else "layer >= 13"


def get_model_ids():
    return [MODEL_FOR_DEBUGGING] if DEBUG else [MODEL_FOR_REAL]


def get_collection_ids():
    # return ["got"] if DEBUG else ["dlk", "repe", "got"]
    return ["got"]
