from typing import Literal

from meta_evals.activations.probe_preparations import EvalInputArrays
from meta_evals.evaluations.predict_from_next_token import (
    NextTokenEvaluation,
    BaseEval,
)
from meta_evals.evaluations.probes.base import BaseProbe
from meta_evals.evaluations.probes.contrast_consistent_search import (
    CcsTrainingConfig,
    train_ccs_probe,
)
from meta_evals.evaluations.probes.difference_in_means import train_dim_probe
from meta_evals.evaluations.probes.linear_artificial_tomography import (
    train_lat_probe,
)
from meta_evals.evaluations.probes.linear_discriminant_analysis import (
    train_lda_probe,
)
from meta_evals.evaluations.probes.logistic_regression import (
    LrConfig,
    train_grouped_lr_probe,
    train_lr_probe,
)
from meta_evals.evaluations.probes.principal_component_analysis import (
    train_grouped_pca_probe,
    train_pca_probe,
)
from meta_evals.evaluations.probes.random import (
    train_random_probe,
    AlwaysIdxProbe,
)
from meta_evals.utils.constants import WORK_WITH_BEHAVIORAL_PROBES, DEBUG

BEHAVIORAL_PREFIX = "behavioral-"

ProbeAlgo = [
    "ccs",
    "lat",
    "dim",
    "lda",
    "lr",
    "lr-g",
    "pca",
    "pca-g",
    "rand",
]
OtherAlgo = [
    "next_tok_prediction",
    "rand_0",
    "rand_1",
]
all_probe_methods = ProbeAlgo + [BEHAVIORAL_PREFIX + algo for algo in ProbeAlgo]
all_algos = (
    all_probe_methods
    + OtherAlgo
    + [BEHAVIORAL_PREFIX + algo for algo in OtherAlgo]
)
EvalMethod = Literal[tuple(all_algos)]

ALL_LATENT_KNOWLEDGE_PROBES: list[EvalMethod] = [
    "ccs",
    "lat",
    "dim",
    "lda",
    "lr",
    "lr-g",
    "pca",
    "pca-g",
]
SUPERVISED_PROBES: list[EvalMethod] = ["dim", "lda", "lr", "lr-g"]
UNSUPERVISED_PROBES: list[EvalMethod] = list(
    set(ALL_LATENT_KNOWLEDGE_PROBES) - set(SUPERVISED_PROBES)
)
GROUPED_PROBES: list[EvalMethod] = ["ccs", "lr-g", "pca-g"]
UNGROUPED_PROBES: list[EvalMethod] = list(
    set(ALL_LATENT_KNOWLEDGE_PROBES) - set(GROUPED_PROBES)
)
ALL_BEHAVIORAL_PROBES: list[EvalMethod] = [
    BEHAVIORAL_PREFIX + "ccs",
    BEHAVIORAL_PREFIX + "lat",
    BEHAVIORAL_PREFIX + "dim",
    BEHAVIORAL_PREFIX + "lda",
    BEHAVIORAL_PREFIX + "lr",
    BEHAVIORAL_PREFIX + "lr-g",
    BEHAVIORAL_PREFIX + "pca",
    BEHAVIORAL_PREFIX + "pca-g",
]


def get_evals_to_work_with():
    evals = (
        ALL_BEHAVIORAL_PROBES
        if WORK_WITH_BEHAVIORAL_PROBES
        else ALL_LATENT_KNOWLEDGE_PROBES
    )
    if DEBUG:
        evals = evals[:4]

    if WORK_WITH_BEHAVIORAL_PROBES:
        evals += [
            # BEHAVIORAL_PREFIX + "next_tok_prediction",
            BEHAVIORAL_PREFIX + "rand",
            BEHAVIORAL_PREFIX + "rand_0",
            BEHAVIORAL_PREFIX + "rand_1",
        ]
    else:
        evals += ["rand"]
    return evals


def train_evaluation(
    eval_method, arrays: EvalInputArrays
) -> BaseProbe | BaseEval | None:
    if eval_method.startswith(BEHAVIORAL_PREFIX):
        eval_method = eval_method[len(BEHAVIORAL_PREFIX) :]

    if eval_method in all_probe_methods:
        return train_probe(eval_method, arrays)
    elif eval_method == "next_tok_prediction":
        return NextTokenEvaluation()
    elif eval_method == "rand_0":
        return AlwaysIdxProbe(0)
    elif eval_method == "rand_1":
        return AlwaysIdxProbe(1)
    else:
        raise ValueError(f"Unknown eval_method: {eval_method}")


def train_probe(
    eval_method: EvalMethod, arrays: EvalInputArrays
) -> BaseProbe | None:

    if eval_method == "ccs":
        if arrays.groups is None:
            return None
        return train_ccs_probe(
            CcsTrainingConfig(),
            activations=arrays.activations,
            groups=arrays.groups,
            answer_types=arrays.answer_types,
            # N.B.: Technically unsupervised!
            labels=arrays.labels,
        )
    elif eval_method == "lat":
        return train_lat_probe(
            activations=arrays.activations,
            answer_types=arrays.answer_types,
        )
    elif eval_method == "dim":
        return train_dim_probe(
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif eval_method == "lda":
        return train_lda_probe(
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif eval_method == "lr":
        return train_lr_probe(
            LrConfig(),
            activations=arrays.activations,
            labels=arrays.labels,
        )
    elif eval_method == "lr-g":
        if arrays.groups is None:
            return None
        return train_grouped_lr_probe(
            LrConfig(),
            activations=arrays.activations,
            groups=arrays.groups,
            labels=arrays.labels,
        )
    elif eval_method == "pca":
        return train_pca_probe(
            activations=arrays.activations,
            answer_types=arrays.answer_types,
        )
    elif eval_method == "pca-g":
        if arrays.groups is None:
            return None
        return train_grouped_pca_probe(
            activations=arrays.activations,
            groups=arrays.groups,
            answer_types=arrays.answer_types,
        )
    elif eval_method == "rand":
        return train_random_probe(
            activations=arrays.activations,
        )
    else:
        raise ValueError(f"Unknown probe_method: {eval_method}")
