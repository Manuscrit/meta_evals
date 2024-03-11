import logging
from dataclasses import dataclass
from typing import Literal, List

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float, Int64
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from meta_evals.activations.inference import ModelT
from meta_evals.datasets.activations.types import ActivationResultRow
from meta_evals.datasets.elk.types import Split
from meta_evals.datasets.elk.utils.filters import DatasetFilter
from meta_evals.models.llms import get_llm_tokenizer
from meta_evals.models.types import LlmId, Llm
from meta_evals.utils.constants import WORK_WITH_BEHAVIORAL_PROBES


@dataclass
class EvalInputArrays:
    activations: Float[np.ndarray, "n d"]  # noqa: F722
    labels: Bool[np.ndarray, "n"]  # noqa: F821
    groups: Int64[np.ndarray, "n"] | None  # noqa: F821
    answer_types: Int64[np.ndarray, "n"] | None  # noqa: F821
    behavioral_labels: Bool[np.ndarray, "n"] | None
    next_step_labels: Float[np.ndarray, "n"] | None


@dataclass
class ActivationArrayDataset:
    rows: list[ActivationResultRow]

    def get(
        self,
        *,
        llm_id: LlmId,
        dataset_filter: DatasetFilter,
        split: Split,
        point_name: str | Literal["logprobs"],
        token_idx: int,
        limit: int | None,
    ) -> EvalInputArrays:
        df_rows = []
        for row in self.rows:
            if (
                dataset_filter.filter(
                    dataset_id=row.dataset_id,
                    answer_type=row.answer_type,
                )
                and row.split == split
                and row.llm_id == llm_id
            ):
                row_dict = dict(
                    label=row.label,
                    group_id=row.group_id,
                    answer_type=row.answer_type,
                    activations=(
                        row.activations[point_name][token_idx].copy()
                        if point_name != "logprobs"
                        else np.array(row.prompt_logprobs)
                    ),
                )
                if WORK_WITH_BEHAVIORAL_PROBES:
                    if len(row.answers_possible) != len(
                        row.labels_possible_answers
                    ):
                        logging.warning(
                            f"'answers_possible' and 'labels_possible_answers' have different lengths.\nSkipping this row ({row.group_id})."
                        )
                        continue
                    (
                        behavioral_label,
                        answer_predicted_from_next_token,
                    ) = compute_behavioral_label(
                        row.next_token_logprobs,
                        row.answers_possible,
                        row.labels_possible_answers,
                        llm_id,
                    )
                    row_dict["behavioral_label"] = behavioral_label
                    row_dict[
                        "answer_predicted_from_next_token"
                    ] = answer_predicted_from_next_token
                    row_dict["non_behavioral_label"] = row_dict["label"]
                    row_dict["label"] = behavioral_label

                df_rows.append(row_dict)
                if limit is not None and len(df_rows) == limit:
                    break
        df = pd.DataFrame(df_rows)

        if df.empty:
            n = 1
        assert not df.empty, (
            llm_id,
            dataset_filter,
            split,
            point_name,
            token_idx,
        )

        group_counts = df["group_id"].value_counts().rename("group_count")
        df = df.join(group_counts, on="group_id")
        df.loc[df["group_count"] <= 1, "group_id"] = np.nan
        if df["group_id"].isna().all():  # type: ignore
            groups = None
        else:
            df = df[df["group_id"].notna()]
            groups = (
                df["group_id"].astype("category").cat.codes.to_numpy()  # type: ignore
            )

        if df["answer_type"].isna().any():  # type: ignore
            answer_types = None
        else:
            answer_types = (
                df["answer_type"]
                .astype("category")
                .cat.codes.to_numpy()  # type: ignore
            )

        labels = df["label"].to_numpy()
        if WORK_WITH_BEHAVIORAL_PROBES:
            behavioral_label = df["behavioral_label"].to_numpy()
            next_step_labels = df["answer_predicted_from_next_token"].to_numpy()
        else:
            behavioral_label = None
            next_step_labels = None

        return EvalInputArrays(
            activations=np.stack(df["activations"].tolist()).astype(np.float32),
            labels=labels,  # type: ignore
            groups=groups,
            answer_types=answer_types,
            behavioral_labels=behavioral_label,
            next_step_labels=next_step_labels,
        )


def compute_behavioral_label(
    next_token_prediction: List,
    possible_answers: List[str],
    labels_per_possible_answer: List[bool],
    llm_id: LlmId,
):
    if not WORK_WITH_BEHAVIORAL_PROBES:
        return None, None

    tokenizer = get_llm_tokenizer(llm_id)
    answer_predicted_from_next_token = get_predicted_label_from_next_token(
        next_token_prediction,
        possible_answers,
        tokenizer,
    )
    if answer_predicted_from_next_token is np.nan:
        return np.nan, np.nan

    assert sum(labels_per_possible_answer) == 1, labels_per_possible_answer
    assert len(possible_answers) == len(labels_per_possible_answer), (
        possible_answers,
        labels_per_possible_answer,
    )
    truthful_answer = possible_answers[labels_per_possible_answer.index(True)]

    if truthful_answer == answer_predicted_from_next_token:
        return True, answer_predicted_from_next_token
    else:
        return False, answer_predicted_from_next_token


def get_predicted_label_from_next_token(
    next_token_prediction: List,
    possible_answers: List[str],
    tokenizer: PreTrainedTokenizerFast,
) -> float:
    # Greedy decoding
    most_likely_token_id = next_token_prediction[0][0]
    decoded_completion = tokenizer.convert_ids_to_tokens([most_likely_token_id])
    assert len(decoded_completion) == 1, decoded_completion
    decoded_completion = decoded_completion[0]

    detected_answers = []
    for answer in possible_answers:
        # TODO REMOVE THIS dirty fix using the strip
        if answer.strip("(").startswith(decoded_completion):
            detected_answers.append(answer)

    if len(detected_answers) > 1:
        logging.warning(
            f"Detected multiple labels for token {decoded_completion}: {detected_answers}"
        )
        return np.nan
    elif len(detected_answers) == 1:
        return detected_answers[0]
    else:
        return np.nan
