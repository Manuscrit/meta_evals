from typing import TypeVar, List

import numpy as np
import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from meta_evals.hooks.grab import grab_many
from meta_evals.models.llms import Llm
from meta_evals.utils.pydantic_ndarray import NdArray

ModelT = TypeVar("ModelT", bound=PreTrainedModel)


class ActivationRow(BaseModel, extra="forbid"):
    text: str
    text_tokenized: list[str]
    activations: dict[str, NdArray]
    token_logprobs: NdArray
    next_token_logprobs: List[tuple[int, float]]

    class Config:
        arbitrary_types_allowed = True


@torch.inference_mode()
def get_model_activations(
    llm: Llm[ModelT, PreTrainedTokenizerFast],
    *,
    text: str,
    last_n_tokens: int | None,
    points_start: int | None,
    points_end: int | None,
    points_skip: int | None,
) -> ActivationRow:
    assert last_n_tokens is None or last_n_tokens > 0, last_n_tokens

    tokens = llm.tokenizer.encode(text, return_tensors="pt")
    assert isinstance(tokens, torch.Tensor)
    tokens = tokens.to(next(llm.model.parameters()).device)
    tokens_str = llm.tokenizer.convert_ids_to_tokens(tokens.squeeze().tolist())
    assert isinstance(tokens_str, list)

    points = llm.points[points_start:points_end:points_skip]
    with grab_many(llm.model, points) as activation_fn:
        output = llm.model.forward(tokens)
        logits: torch.Tensor = output.logits
        layer_activations = activation_fn()

    logprobs = logits.log_softmax(dim=-1)
    logprobs_shifted = logprobs[0, :-1]
    tokens_shifted = tokens[0, 1:, None]
    token_logprobs = (
        logprobs_shifted.gather(dim=-1, index=tokens_shifted)
        .squeeze(0)
        .detach()
    )
    next_token_logprobs = logprobs[0, -1].detach()
    filtered_indices, filtered_logprobs = filter_top_p(
        next_token_logprobs, top_p=0.95
    )
    filtered_indices = filtered_indices.cpu().numpy()
    filtered_logprobs = filtered_logprobs.cpu().numpy()
    filtered_next_token_logprobs = [
        (idx, logprob)
        for idx, logprob in zip(filtered_indices, filtered_logprobs)
    ]

    def get_activation(activations: torch.Tensor) -> np.ndarray:
        activations = activations.squeeze(0)
        if last_n_tokens is not None:
            activations = activations[-last_n_tokens:]
        activations = activations.detach()
        # bfloat16 is not supported by numpy. We lose some precision by converting to
        # float16, but it significantly saves space an empirically makes little
        # difference.
        activations = activations.to(dtype=torch.float16)
        return activations.cpu().numpy()

    return ActivationRow(
        text=text,
        text_tokenized=tokens_str,
        activations={
            name: get_activation(activations)
            for name, activations in layer_activations.items()
        },
        token_logprobs=token_logprobs.float().cpu().numpy(),
        next_token_logprobs=filtered_next_token_logprobs,
    )


def filter_top_p(next_token_logprobs, top_p: float):
    """
    Filter the next_token_logprobs to only include tokens within the top_p cumulative probability.

    Parameters:
    - next_token_logprobs (torch.Tensor): The log probabilities of the next tokens, shape (num_tokens,)
    - top_p (float): The cumulative probability threshold to include. Value between 0 and 1.

    Returns:
    - filtered_ids (torch.Tensor): The token IDs within the top_p cumulative probability.
    - filtered_logprobs (torch.Tensor): The log probabilities of the filtered token IDs.
    """
    # Sort the log probabilities in descending order and also keep the original indices (token IDs)
    sorted_logprobs, sorted_indices = torch.sort(
        next_token_logprobs, descending=True
    )

    # Convert log probabilities to probabilities for the calculation of the cumulative sum
    sorted_probs = torch.exp(sorted_logprobs)

    # Calculate the cumulative probabilities
    cum_probs = torch.cumsum(sorted_probs, dim=0)

    # Find the index where the cumulative probability exceeds top_p
    # This includes one additional element beyond the threshold, ensuring the sum exceeds top_p
    cutoff_index = torch.searchsorted(cum_probs, top_p, right=True) + 1

    # Filter the sorted arrays to keep only the top_p elements
    filtered_indices = sorted_indices[:cutoff_index]
    filtered_logprobs = sorted_logprobs[:cutoff_index]

    return filtered_indices, filtered_logprobs
