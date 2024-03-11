from typing import Literal, get_args, List

from pydantic import BaseModel

Split = Literal["train", "train-hparams", "validation"]

DatasetId = Literal[
    "arc_challenge",
    "arc_easy",
    "got_cities",
    "got_sp_en_trans",
    "got_larger_than",
    "got_cities_cities_conj",
    "got_cities_cities_disj",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
    "truthful_model_written",
    "true_false",
    "imdb",
    "imdb/simple",
    "amazon_polarity",
    "ag_news",
    "dbpedia_14",
    "rte",
    "copa",
    "boolq",
    "boolq/simple",
    "piqa",
    "open_book_qa/simple",
    "race/simple",
    "arc_challenge/simple",
    "arc_easy/simple",
    "common_sense_qa/simple",
    "persona",
    "persona.desire-for-acquiring-power",
]


DlkDatasetId = Literal[
    "imdb",
    "imdb/simple",
    "amazon_polarity",
    "ag_news",
    "dbpedia_14",
    "rte",
    "copa",
    "boolq",
    "boolq/simple",
    "piqa",
]

GroupedDatasetId = Literal[
    "arc_challenge",
    "arc_easy",
    "common_sense_qa",
    "open_book_qa",
    "race",
    "truthful_qa",
    "truthful_model_written",
    "true_false",
]

TemplateType = Literal["repe", "dlk", "simple"]


class Row(BaseModel, extra="forbid"):
    dataset_id: DatasetId
    split: Split
    text: str | None = None
    label: bool | None = None
    format_args: dict[str, str] | None = None
    group_id: str | None = None
    """
    Rows are grouped, for example by question, in order to allow for probes that take
    into account intra-group relationships.
    """
    answer_type: str | None = None
    """
    For example, 'true' and 'false' for answers to true/false questions, or 'A', 'B',
    'C', or 'D' for multiple choice questions.
    If not set, the prompt template doesn't include any consistent answer templates
    (e.g. it's just question-answer).
    """
    # To support multiple choice questions
    answers_possible: List[str] | None = None
    labels_possible_answers: List[bool] | None = None


def is_dataset_grouped(dataset_id: DatasetId) -> bool:
    return dataset_id in get_args(GroupedDatasetId)
