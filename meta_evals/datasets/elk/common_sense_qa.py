from typing import Any

import datasets

from meta_evals.datasets.elk.types import (
    Row,
    DatasetId,
    Split,
    TemplateType,
)
from meta_evals.datasets.utils.shuffles import deterministic_shuffle
from meta_evals.datasets.utils.splits import split_train

_DATASET_IDS: dict[TemplateType, DatasetId] = {
    "repe": "common_sense_qa",
    "simple": "common_sense_qa/simple",
}
# Taken from https://arxiv.org/abs/2310.01405 D.1.6.
_TEMPLATES: dict[TemplateType, str] = {
    "repe": (
        "Based on commonsense reasoning, consider the plausibility of the\n"
        "answer to the following question:\n"
        "Question: {question}\n"
        "Answer: {answer}\n"
        "The probability of the answer being plausible is"
    ),
    "simple": "Question: {question}\nAnswer: {answer}",
}


def get_common_sense_qa(template_type: TemplateType) -> dict[str, Row]:
    return {
        **_get_common_sense_qa_split("train", template_type=template_type),
        **_get_common_sense_qa_split("validation", template_type=template_type),
    }


def _get_common_sense_qa_split(
    split: Split, template_type: TemplateType
) -> dict[str, Row]:
    dataset_id = _DATASET_IDS[template_type]
    template = _TEMPLATES[template_type]
    dataset: Any = datasets.load_dataset("commonsense_qa")
    results = {}
    for row in deterministic_shuffle(dataset[split], lambda row: row["id"]):
        group_id = row["id"]
        for choice, choice_label in zip(
            row["choices"]["text"], row["choices"]["label"], strict=True
        ):
            format_args = dict(question=row["question"], answer=choice)
            results[f"{dataset_id}-{group_id}-{choice_label}"] = Row(
                dataset_id=dataset_id,
                split=split_train(
                    split, seed="common_sense_qa", row_id=group_id
                ),
                group_id=group_id,
                text=template.format(**format_args),
                label=row["answerKey"] == choice_label,
                format_args=format_args,
            )
    return results
