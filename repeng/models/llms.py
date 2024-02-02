from typing import Any, cast, get_args, overload

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from repeng.models import points
from repeng.models.types import Gpt2Id, Llama2Id, Llm, LlmId, PythiaDpoId, PythiaId


@overload
def get_llm(
    llm_id: PythiaId | PythiaDpoId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    ...


@overload
def get_llm(
    llm_id: Gpt2Id,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    ...


@overload
def get_llm(
    llm_id: Llama2Id,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[LlamaForCausalLM, PreTrainedTokenizerFast]:
    ...


def get_llm(
    llm_id: LlmId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[Any, Any]:
    if llm_id in get_args(PythiaId):
        return pythia(cast(PythiaId, llm_id), device, use_half_precision)
    elif llm_id in get_args(Gpt2Id):
        return gpt2(device, use_half_precision)
    elif llm_id in get_args(Llama2Id):
        return llama2(cast(Llama2Id, llm_id), device, use_half_precision)
    elif llm_id in get_args(PythiaDpoId):
        return pythia_dpo(cast(PythiaDpoId, llm_id), device, use_half_precision)
    else:
        raise ValueError(f"Unknown LLM ID: {llm_id}")


def gpt2(
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GPT2LMHeadModel, PreTrainedTokenizerFast]:
    dtype = torch.float16 if use_half_precision else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", device_map=device, torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        points.gpt2(),
    )


def pythia(
    pythia_id: PythiaId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    dtype = torch.float16 if use_half_precision else torch.float32
    model = GPTNeoXForCausalLM.from_pretrained(
        f"EleutherAI/{pythia_id}",
        device_map=device,
        torch_dtype=dtype,
    )
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{pythia_id}")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        points.pythia(pythia_id),
    )


def pythia_dpo(
    pythia_dpo_id: PythiaDpoId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GPTNeoXForCausalLM, PreTrainedTokenizerFast]:
    dtype = torch.float16 if use_half_precision else torch.float32
    if pythia_dpo_id == "pythia-dpo-1b":
        model_id = "Leogrin/eleuther-pythia1b-hh-dpo"
        pythia_id = "pythia-1b"
    elif pythia_dpo_id == "pythia-sft-1b":
        model_id = "Leogrin/eleuther-pythia1b-hh-sft"
        pythia_id = "pythia-1b"
    elif pythia_dpo_id == "pythia-dpo-1.4b":
        model_id = "Leogrin/eleuther-pythia1.4b-hh-dpo"
        pythia_id = "pythia-1.4b"
    elif pythia_dpo_id == "pythia-sft-1.4b":
        model_id = "Leogrin/eleuther-pythia1.4b-hh-sft"
        pythia_id = "pythia-1.4b"
    else:
        raise ValueError(f"Unknown Pythia DPO ID: {pythia_dpo_id}")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(model, tokenizer, points.pythia(pythia_id))


def llama2(
    llama_id: Llama2Id,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[LlamaForCausalLM, PreTrainedTokenizerFast]:
    dtype = torch.bfloat16 if use_half_precision else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        f"meta-llama/{llama_id}", device_map=device, torch_dtype=dtype
    )
    assert isinstance(model, LlamaForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{llama_id}")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(
        model,
        tokenizer,
        points.llama2(llama_id),
    )
