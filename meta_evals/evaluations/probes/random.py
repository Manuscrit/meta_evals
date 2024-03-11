"""
Baseline random probe.
"""
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from overrides import override

from meta_evals.evaluations.probes.base import (
    DotProductProbe,
    BaseProbe,
    PredictResult,
)


def train_random_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
) -> DotProductProbe:
    _, hidden_dim = activations.shape
    probe = np.random.uniform(-1, 1, size=hidden_dim)
    probe /= np.linalg.norm(probe)
    return DotProductProbe(probe=probe)


@dataclass
class AlwaysIdxProbe(BaseProbe):
    probe: Float[np.ndarray, "d"]  # noqa: F821
    is_flippable: bool = False

    def __init__(self, idx_to_return: int):
        self.idx_to_return = float(idx_to_return)

    @override
    def predict(
        self,
        activations: Float[np.ndarray, "n d"],  # noqa: F722
    ) -> PredictResult:
        len_activations = activations.shape[0]
        if self.idx_to_return == 0:
            logits = np.ones(len_activations) * 5
        elif self.idx_to_return == 1:
            logits = -np.ones(len_activations) * 5
        else:
            raise ValueError(f"Invalid idx_to_return: {self.idx_to_return}")
        return PredictResult(logits=logits)
