from datetime import datetime

from meta_evals.datasets.activations.creation import create_activations_dataset
from meta_evals.datasets.elk.utils.limits import Limits, SplitLimits
from meta_evals.utils.constants import (
    DEBUG,
    MODEL_FOR_DEBUGGING,
    MODEL_FOR_REAL,
    FASTER_RUN,
    DEBUG_VERSION,
    get_number_sample_debugging,
    inital_layer,
    layer_skip,
)
from meta_evals.utils.utils import (
    get_torch_device,
    get_dataset_ids,
)

"""
This memory usage is neglecting the size of the next_token_logprobs.

DEBUG
~6 datasets
 * 2 layers
 * 1 token
 * 300 questions
 * 3 answers
 * 512 hidden dim size
 * 2 bytes
= >0.5GB

VANILLA
18 datasets
 * 20 layers
 * 1 token
 * 4400 (400 + 2000 + 2000) questions
 * 3 answers
 * 5120 hidden dim size
 * 2 bytes
= 49GB
"""


if __name__ == "__main__":

    dataset_ids = get_dataset_ids()
    create_activations_dataset(
        tag=f"debug_{DEBUG_VERSION}"
        if DEBUG
        else f"datasets_{datetime.now().isoformat()}",
        # else "datasets_2024-03-05T08:37:13.825450",
        # else "datasets_2024-03-05T10:23:50.961381",
        llm_ids=[MODEL_FOR_DEBUGGING if DEBUG else MODEL_FOR_REAL],
        dataset_ids=dataset_ids,
        group_limits=Limits(
            default=SplitLimits(
                train=get_number_sample_debugging() if DEBUG else 400,
                train_hparams=get_number_sample_debugging()
                if DEBUG
                else (400 if FASTER_RUN else 2000),
                validation=get_number_sample_debugging()
                if DEBUG
                else (400 if FASTER_RUN else 2000),
            ),
            by_dataset={
                "truthful_qa": SplitLimits(
                    train=0,
                    train_hparams=0,
                    validation=get_number_sample_debugging()
                    if DEBUG
                    else (400 if FASTER_RUN else 2000),
                ),
                "persona.desire-for-acquiring-power": SplitLimits(
                    train=0,
                    train_hparams=0,
                    validation=get_number_sample_debugging()
                    if DEBUG
                    else (400 if FASTER_RUN else 2000),
                ),
            },
        ),
        num_tokens_from_end=1,
        device=get_torch_device(),
        layers_start=inital_layer(),
        layers_end=None,
        layers_skip=layer_skip(),
    )
