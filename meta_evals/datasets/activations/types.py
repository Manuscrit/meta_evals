from pydantic import BaseModel

from meta_evals.datasets.elk.types import DatasetId, Split
from meta_evals.models.llms import LlmId
from meta_evals.utils.pydantic_ndarray import NdArray


class ActivationResultRow(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    group_id: str | None
    answer_type: str | None
    activations: dict[str, NdArray]  # (s, d)
    prompt_logprobs: float
    label: bool
    split: Split
    llm_id: LlmId

    class Config:
        arbitrary_types_allowed = True
