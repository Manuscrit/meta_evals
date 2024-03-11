from pathlib import Path

import torch
from dotenv import load_dotenv
from mppr import MContext, MDict
from pydantic import BaseModel

from meta_evals.activations.inference import get_model_activations
from meta_evals.datasets.activations.types import ActivationResultRow
from meta_evals.datasets.elk.types import Row, DatasetId
from meta_evals.datasets.elk.utils.fns import get_dataset
from meta_evals.datasets.elk.utils.limits import Limits, limit_groups
from meta_evals.models.llms import LlmId
from meta_evals.models.loading import load_llm_oioo
from meta_evals.utils.constants import ACTIVATION_DIR, DEBUG, FASTER_RUN
from meta_evals.utils.utils import check_for_mps

assert load_dotenv()


class _RowWithLlm(Row):
    llm_id: LlmId


class _Dataset(BaseModel, extra="forbid"):
    rows: dict[str, Row]


def create_activations_dataset(
    tag: str,
    llm_ids: list[LlmId],
    dataset_ids: list[DatasetId],
    group_limits: Limits,
    device: torch.device,
    num_tokens_from_end: int | None,
    layers_start: int | None,
    layers_end: int | None,
    layers_skip: int | None,
) -> list[ActivationResultRow]:
    mcontext = MContext(Path("output/create-activations-dataset"))
    dataset_ids_mdict: MDict[DatasetId] = mcontext.create(
        {dataset_id: dataset_id for dataset_id in dataset_ids},
    )
    inputs = (
        dataset_ids_mdict.map_cached(
            f"datasets_{tag}",
            lambda _, dataset_id: _Dataset(rows=get_dataset(dataset_id)),
            to=_Dataset,
        )
        .flat_map(
            lambda _, dataset: {key: row for key, row in dataset.rows.items()}
        )
        .filter(limit_groups(group_limits))
        .flat_map(
            lambda key, row: {
                f"{key}-{llm_id}": _RowWithLlm(
                    **row.model_dump(), llm_id=llm_id
                )
                for llm_id in llm_ids
            }
        )
        # avoids constantly reloading models with OIOO
        .sort(lambda _, row: row.llm_id)
    )
    print(
        "labels",
        set([dict(dict_)["label"] for dict_ in list(inputs.values.values())]),
    )
    return (
        inputs.map_cached(
            f"activations_{tag}",
            fn=lambda _, value: get_model_activations(
                load_llm_oioo(
                    value.llm_id,
                    device=device,
                    use_half_precision=False if check_for_mps() else True,
                ),
                text=value.text,
                last_n_tokens=num_tokens_from_end,
                points_start=layers_start,
                points_end=layers_end,
                points_skip=layers_skip,
            ),
            to="pickle",
        )
        .join(
            inputs,
            lambda _, activations, input: ActivationResultRow(
                dataset_id=input.dataset_id,
                split=input.split,
                answer_type=input.answer_type,
                label=input.label,
                activations=activations.activations,
                prompt_logprobs=activations.token_logprobs.sum().item(),
                next_token_logprobs=activations.next_token_logprobs,
                group_id=input.group_id,
                llm_id=input.llm_id,
                answers_possible=input.answers_possible,
                labels_possible_answers=input.labels_possible_answers,
                prompt=input.text,
            ),
        )
        .upload(
            f"{ACTIVATION_DIR}/{tag}"
            if DEBUG or FASTER_RUN
            else f"s3://repeng/datasets/activations/{tag}.pickle",
            to="pickle",
        )
    ).get()
