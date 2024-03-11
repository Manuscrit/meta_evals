import numpy as np
from jaxtyping import Bool, Float, Int64

from meta_evals.evals_utils.logits import (
    eval_logits_by_question,
    eval_logits_by_row,
)
from meta_evals.evals_utils.types import QuestionsEvalResult, RowsEvalResult
from meta_evals.evaluations.predict_from_next_token import (
    PreviousTokenEvaluation,
    NextTokenEvaluation,
)
from meta_evals.evaluations.probes.base import BaseProbe


def eval_probe_by_row(
    probe: BaseProbe,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
) -> RowsEvalResult:
    result = probe.predict(activations)
    eval_result = eval_logits_by_row(
        logits=result.logits,
        labels=labels,
    )
    if probe.is_flippable:
        eval_result_flipped = eval_logits_by_row(
            logits=-result.logits,
            labels=labels,
        )
        if eval_result.roc_auc_score < eval_result_flipped.roc_auc_score:
            return eval_result_flipped.model_copy(
                update=dict(is_flipped=True),
            )
        else:
            return eval_result
    else:
        return eval_result


def eval_probe_by_question(
    probe: BaseProbe,
    *,
    activations: Float[np.ndarray, "n d"],  # noqa: F722
    groups: Int64[np.ndarray, "n d"],  # noqa: F722
    labels: Bool[np.ndarray, "n"],  # noqa: F821
    allow_flipping: bool = True,
    **kwargs,
) -> QuestionsEvalResult:
    if isinstance(probe, NextTokenEvaluation):
        result = probe.predict(
            llm_id=kwargs["llm_id"],
            labels_from_next_token_predictions=kwargs[
                "labels_from_next_token_predictions"
            ],
        )
    elif isinstance(probe, PreviousTokenEvaluation):
        result = probe.predict(
            llm_id=kwargs["llm_id"],
            dataset_id=kwargs["dataset_id"],
            prompt_logprobs_seq=kwargs["prompt_logprobs_seq"],
            prompt_tokens=kwargs["prompt_tokens"],
        )
    else:
        result = probe.predict(activations)

    eval_result = eval_logits_by_question(
        logits=result.logits,
        labels=labels,
        groups=groups,
    )
    if allow_flipping:
        eval_result_flipped = eval_logits_by_question(
            logits=-result.logits,
            labels=labels,
            groups=groups,
        )
        if eval_result.accuracy < eval_result_flipped.accuracy:
            return QuestionsEvalResult(
                accuracy=eval_result_flipped.accuracy,
                is_flipped=True,
                n=eval_result.n,
            )
    return eval_result
