from dataclasses import dataclass

from pydantic import BaseModel

from meta_evals.datasets.elk.utils.filters import DatasetFilter
from meta_evals.models.types import LlmId
from meta_evals.evaluations.probes.base import BaseProbe
from meta_evals.evaluations.probes.collections import EvalMethod


@dataclass
class TrainSpec:
    llm_id: LlmId
    dataset: DatasetFilter
    probe_method: EvalMethod
    point_name: str
    token_idx: int


@dataclass
class EvalSpec:
    train_spec: TrainSpec
    probe: BaseProbe
    dataset: DatasetFilter


@dataclass
class EvalResult:
    accuracy: float
    n: int


class PipelineResultRow(BaseModel, extra="forbid"):
    llm_id: LlmId
    train_dataset: str
    eval_dataset: str
    probe_method: EvalMethod
    point_name: str
    token_idx: int
    accuracy: float
    accuracy_n: int
    accuracy_hparams: float
    accuracy_hparams_n: float
