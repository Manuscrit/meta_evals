from abc import abstractmethod, ABC
from dataclasses import dataclass
from jaxtyping import Float, Int64
from transformers import LlamaTokenizerFast
from typing_extensions import override

import numpy as np

from meta_evals.activations.probe_preparations import (
    get_predicted_label_from_next_token,
)
from meta_evals.evaluations.probes.base import BaseProbe, PredictResult
from meta_evals.models.llms import get_llm_tokenizer


# class BaseEval(ABC):
#     @abstractmethod
#     def predict(
#         self,
#         llm_id: str,
#         labels_from_next_token_predictions: Float[np.ndarray, "n"],
#     ) -> "PredictResult":
#         """
#         Predicts the probability of the label being true for each row.
#         """
#         ...
#


@dataclass
class NextTokenEvaluation:
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


class PreviousTokenEvaluation:
    @override
    def predict(
        self,
        llm_id: str,
        dataset_id: str,
        prompt_logprobs_seq: Float[np.ndarray, "n s"],
        prompt_tokens: Int64[np.ndarray, "n s"],
    ) -> PredictResult:
        tokenizer = get_llm_tokenizer(llm_id)
        possible_tokens_of_interest = get_token_of_interest(
            dataset_id, tokenizer
        )
        # already_found = False
        idx = np.zeros(len(prompt_tokens), dtype=int)
        prompt_tokens = [np.array(seq) for seq in prompt_tokens]
        for token_of_interest in possible_tokens_of_interest:
            last_token = token_of_interest[-1]
            # str_last_token = tokenizer.decode([last_token])
            for i, prompt_i_tokens in enumerate(prompt_tokens):
                positions = np.where(prompt_i_tokens == last_token)
                assert (
                    len(positions) <= 1
                ), f"Expected 1 position, but got {len(positions)}."
                if len(positions[0]) > 0:
                    assert idx[i] == 0, "Multiple tokens of interest found."
                    if isinstance(tokenizer, LlamaTokenizerFast):
                        # Because the tokenizer output a start of string token that does have a log prob
                        assert len(prompt_i_tokens) - 1 == len(
                            prompt_logprobs_seq[i]
                        )
                        idx[i] = positions[0][0] - 1
                    else:
                        raise NotImplementedError(
                            "Only LlamaTokenizerFast is supported."
                        )

        assert np.all(idx != 0), "Not all tokens of interest found."

        logits = [
            log_probs_seq[idx[i]]
            for i, log_probs_seq in enumerate(prompt_logprobs_seq)
        ]
        logits = np.array(logits)
        return PredictResult(logits=logits)


def get_token_of_interest(dataset_id: str, tokenizer) -> int:
    if "persona." in dataset_id:
        possible_strings_of_interest = ["\nAnswer: Yes", "\nAnswer: No"]
        # possible_strings_of_interest = ["Yes", "No"]
    else:
        raise NotImplementedError(f"Dataset {dataset_id} not supported.")

    tokens_of_interest = [
        tokenizer.encode(string_of_interest)
        for string_of_interest in possible_strings_of_interest
    ]
    # tokens_of_interest = [toks[-1] for toks in tokens_of_interest]
    return tokens_of_interest
