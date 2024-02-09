"""
Replication of difference-in-means probes.

See DIM probes in <https://arxiv.org/abs/2312.01037v1> and MMP described in
<https://arxiv.org/abs/2310.06824>.
"""

import numpy as np
from jaxtyping import Bool, Float

from repeng.probes.base import DotProductProbe


def train_dim_probe(
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> DotProductProbe:
    return DotProductProbe(
        activations[labels].mean(axis=0) - activations[~labels].mean(axis=0)
    )