import os
import pathlib

DEBUG = True
DEBUG_VERSION = "v0.30"
# MODEL_FOR_DEBUGGING = "pythia-70m"
MODEL_FOR_DEBUGGING = "Mistral-7B-Instruct"
# MODEL_FOR_DEBUGGING = "gemma-2b-it"
MODEL_FOR_REAL = (
    # "Llama-2-13b-chat-hf"
    "Mistral-7B-Instruct"
    # "gemma-2b-it"
)
FASTER_RUN = True
WORK_WITH_BEHAVIORAL_PROBES = False
USE_MPS = True
WORK_WITH_NEXT_TOK_ALGO = False
WORK_WITH_PREVIOUS_TOK_ALGO = True

REPO_ROOT = pathlib.Path(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PLOT_DIR = REPO_ROOT / "data" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
ACTIVATION_DIR = REPO_ROOT / "data" / "activations"
ACTIVATION_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 2024
PERSONA_REPO = pathlib.Path("/Users/maximeriche/Dev/anthropic_evals")
PERSONAS_TO_USE = [
    "persona.desire-for-acquiring-power",
    "persona.desire-for-self-improvement",
    "persona.machiavellianism",
    "persona.no-shut-down",
    "persona.agreeableness",
]


def get_layer_filter():
    return f"layer >= {max(15, inital_layer())}"


def inital_layer():
    # return 1 if DEBUG else 13
    return 1


def layer_skip():
    # return 8 if FASTER_RUN or DEBUG else 2
    return 2


def get_model_ids():
    return [MODEL_FOR_DEBUGGING] if DEBUG else [MODEL_FOR_REAL]


def get_ds_collection_ids():
    # return ["got"] if DEBUG or FASTER_RUN else ["dlk", "repe", "got"]
    return ["dlk", "repe", "got"]


def get_number_sample_debugging():
    assert DEBUG
    return 100 if USE_MPS else 10
