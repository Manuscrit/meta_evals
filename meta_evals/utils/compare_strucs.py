from dataclasses import dataclass

from pydantic import BaseModel

from meta_evals.datasets.elk.utils.filters import DatasetFilter
from meta_evals.models.types import LlmId
from meta_evals.probes.base import BaseProbe
from meta_evals.probes.collections import ProbeMethod


@dataclass
class TrainSpec:
    llm_id: LlmId
    dataset: DatasetFilter
    probe_method: ProbeMethod
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
    probe_method: ProbeMethod
    point_name: str
    token_idx: int
    accuracy: float
    accuracy_n: int
    accuracy_hparams: float
    accuracy_hparams_n: float
