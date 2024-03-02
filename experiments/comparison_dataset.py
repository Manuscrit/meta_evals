import torch

from meta_evals.datasets.activations.creation import create_activations_dataset
from meta_evals.datasets.elk.utils.collections import (
    DatasetCollectionId,
    resolve_dataset_ids,
)
from meta_evals.datasets.elk.utils.limits import Limits, SplitLimits
from meta_evals.utils.constants import DEBUG, MODEL_FOR_DEBUGGING
from meta_evals.utils.utils import check_for_mps

"""
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

    collections: list[DatasetCollectionId] = (
        ["got"] if DEBUG else ["dlk", "repe", "got"]
    )
    create_activations_dataset(
        tag="debug" if DEBUG else "datasets_2024-02-23_truthfulqa_v1",
        llm_ids=[MODEL_FOR_DEBUGGING if DEBUG else "Llama-2-13b-chat-hf"],
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
                train=4 if DEBUG else 400,
                train_hparams=20 if DEBUG else 2000,
                validation=20 if DEBUG else 2000,
            ),
            by_dataset={
                "truthful_qa": SplitLimits(
                    train=0,
                    train_hparams=0,
                    validation=20 if DEBUG else 2000,
                )
            },
        ),
        num_tokens_from_end=1,
        device=torch.device("mps") if check_for_mps() else torch.device("cuda"),
        layers_start=1,
        layers_end=None,
        layers_skip=2,
    )
