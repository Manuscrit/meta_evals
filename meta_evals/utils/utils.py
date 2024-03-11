import torch

from meta_evals.datasets.elk.utils.collections import resolve_dataset_ids
from meta_evals.utils.constants import (
    USE_MPS,
    WORK_WITH_BEHAVIORAL_PROBES,
    get_ds_collection_ids,
    DEBUG,
)


def check_for_mps():
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"Built: {built}, Available: {available}")
    return built and available


def get_torch_device():
    return (
        (torch.device("mps") if USE_MPS else torch.device("cpu"))
        if check_for_mps()
        else torch.device("cuda")
    )


def get_dataset_ids(include_validation_ds_only: bool = True):
    # if WORK_WITH_BEHAVIORAL_PROBES:
    #     return ["truthful_qa"]

    collections = get_ds_collection_ids()
    dataset_ids = [
        *[
            dataset_id
            for collection in collections
            for dataset_id in resolve_dataset_ids(collection)
        ],
    ]
    if DEBUG:
        dataset_ids = dataset_ids[:1]

    if include_validation_ds_only:
        dataset_ids.extend(
            [
                "truthful_qa",
                "persona.desire-for-acquiring-power",
            ]
        )
    return dataset_ids
