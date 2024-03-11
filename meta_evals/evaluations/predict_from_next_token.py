from abc import abstractmethod, ABC
from dataclasses import dataclass
from jaxtyping import Float, Int64
from typing_extensions import override

import numpy as np

from meta_evals.activations.probe_preparations import (
    get_predicted_label_from_next_token,
)
from meta_evals.evaluations.probes.base import BaseProbe, PredictResult
from meta_evals.models.llms import get_llm_tokenizer


class BaseEval(ABC):
    @abstractmethod
    def predict(
        self,
        llm_id: str,
        labels_from_next_token_predictions: Float[np.ndarray, "n"],
    ) -> "PredictResult":
        """
        Predicts the probability of the label being true for each row.
        """
        ...


@dataclass
class NextTokenEvaluation(BaseEval):
    @override
    def predict(
        self,
        llm_id: str,
        labels_from_next_token_predictions: Float[np.ndarray, "n"],
    ) -> PredictResult:
        logits = convert_bool_to_logits(labels_from_next_token_predictions)
        return PredictResult(logits=logits)


def convert_bool_to_logits(
    labels: np.ndarray,
) -> np.ndarray:
    logits = (labels.astype(float) * 2 - 1) * 10
    return logits
