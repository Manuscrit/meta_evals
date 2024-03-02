from datetime import datetime

import torch

from meta_evals.datasets.activations.creation import create_activations_dataset
from meta_evals.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from meta_evals.datasets.elk.utils.limits import Limits, SplitLimits
from meta_evals.utils.constants import (
    DEBUG,
    MODEL_FOR_DEBUGGING,
    MODEL_FOR_REAL,
    get_collection_ids,
)
from meta_evals.utils.utils import check_for_mps


"""
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

    collections: list[DatasetCollectionId] = get_collection_ids()
    create_activations_dataset(
        tag="debug_v0.2" if DEBUG else f"datasets_{datetime.now().isoformat()}",
        llm_ids=[MODEL_FOR_DEBUGGING if DEBUG else MODEL_FOR_REAL],
        dataset_ids=[
            *[
                dataset_id
                for collection in collections
                for dataset_id in resolve_dataset_ids(collection)
            ],
            "truthful_qa",
        ],
        group_limits=Limits(
            default=SplitLimits(
                train=100 if DEBUG else 400,
                train_hparams=100 if DEBUG else 2000,
                validation=100 if DEBUG else 2000,
            ),
            by_dataset={
                "truthful_qa": SplitLimits(
                    train=0,
                    train_hparams=0,
                    validation=100 if DEBUG else 2000,
                )
            },
        ),
        num_tokens_from_end=1,
        device=torch.device("mps") if check_for_mps() else torch.device("cuda"),
        # layers_start=13,
        layers_start=1 if DEBUG else 13,
        layers_end=None,
        # layers_skip=2,
        layers_skip=4,
    )
