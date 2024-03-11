from typing import List

from pydantic import BaseModel

from meta_evals.datasets.elk.types import DatasetId, Split
from meta_evals.models.llms import LlmId
from meta_evals.utils.pydantic_ndarray import NdArray


class ActivationResultRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    group_id: str | None
    answer_type: str | None
    activations: dict[str, NdArray]  # (s, d)
    prompt: str
    prompt_logprobs: float
    label: bool | None
    split: Split
    llm_id: LlmId
    # For multi question answering and behavioral probes
    next_token_logprobs: List[tuple[int, float]] | None
    answers_possible: List[str] | None
    labels_possible_answers: List[bool] | None

    class Config:
        arbitrary_types_allowed = True
