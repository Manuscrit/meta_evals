from typing import Any, cast, get_args, overload

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GemmaForCausalLM,
    GPT2LMHeadModel,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PreTrainedTokenizerFast,
)

from meta_evals.models import points
from meta_evals.models.types import (
    GemmaId,
    Gpt2Id,
    Llama2Id,
    Llm,
    LlmId,
    MistralId,
    PythiaDpoId,
    PythiaId,
)
from meta_evals.utils.constants import USE_MPS
from meta_evals.utils.utils import check_for_mps

_MISTRAL_HF_IDS: dict[MistralId, str] = {
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
}


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


@overload
def get_llm(
    llm_id: MistralId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[MistralForCausalLM, PreTrainedTokenizerFast]:
    ...


@overload
def get_llm(
    llm_id: GemmaId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GemmaForCausalLM, PreTrainedTokenizerFast]:
    ...


def get_llm(
    llm_id: LlmId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[Any, Any]:

    if check_for_mps() and USE_MPS:
        assert device.type == "mps"
        assert not use_half_precision

    if llm_id in get_args(PythiaId):
        return pythia(cast(PythiaId, llm_id), device, use_half_precision)
    elif llm_id in get_args(Gpt2Id):
        return gpt2(device, use_half_precision)
    elif llm_id in get_args(Llama2Id):
        return llama2(cast(Llama2Id, llm_id), device, use_half_precision)
    elif llm_id in get_args(PythiaDpoId):
        return pythia_dpo(cast(PythiaDpoId, llm_id), device, use_half_precision)
    elif llm_id in get_args(MistralId):
        return mistral(cast(MistralId, llm_id), device, use_half_precision)
    elif llm_id in get_args(GemmaId):
        return gemma(cast(GemmaId, llm_id), device, use_half_precision)
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
    if check_for_mps() and USE_MPS:
        assert device.type == "mps"
        use_half_precision = False
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
    model_id, pythia_id = get_pythia_dpo_ids(pythia_dpo_id)
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, device_map=device, torch_dtype=dtype
    )
    assert isinstance(model, GPTNeoXForCausalLM)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    return Llm(model, tokenizer, points.pythia(pythia_id))


def get_pythia_dpo_ids(pythia_dpo_id) -> list[PythiaDpoId]:
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
    return model_id, pythia_id


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


def mistral(
    mistral_id: MistralId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[MistralForCausalLM, PreTrainedTokenizerFast]:
    dtype = torch.bfloat16 if use_half_precision else torch.float32
    hf_id = _MISTRAL_HF_IDS[mistral_id]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map=device, torch_dtype=dtype
    )
    assert isinstance(model, MistralForCausalLM)
    return Llm(
        model,
        tokenizer,
        points.mistral(mistral_id),
    )


def gemma(
    gemma_id: GemmaId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[GemmaForCausalLM, PreTrainedTokenizerFast]:
    dtype = torch.bfloat16 if use_half_precision else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(f"google/{gemma_id}")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    model = AutoModelForCausalLM.from_pretrained(
        f"google/{gemma_id}", device_map=device, torch_dtype=dtype
    )
    assert isinstance(model, GemmaForCausalLM)
    return Llm(
        model,
        tokenizer,
        points.gemma(gemma_id),
    )


TOKENIZER_LOADED: dict[LlmId, PreTrainedTokenizerFast] = {}


def get_llm_tokenizer(
    llm_id: LlmId,
) -> PreTrainedTokenizerFast:
    if llm_id in TOKENIZER_LOADED:
        return TOKENIZER_LOADED[llm_id]

    if llm_id in get_args(PythiaId):
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{llm_id}")
    elif llm_id in get_args(Gpt2Id):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif llm_id in get_args(Llama2Id):
        tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{llm_id}")
    elif llm_id in get_args(PythiaDpoId):
        model_id, pythia_id = get_pythia_dpo_ids(llm_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    elif llm_id in get_args(MistralId):
        tokenizer = AutoTokenizer.from_pretrained(_MISTRAL_HF_IDS[llm_id])
    elif llm_id in get_args(GemmaId):
        tokenizer = AutoTokenizer.from_pretrained(f"google/{llm_id}")
    else:
        raise ValueError(f"Unknown LLM ID: {llm_id}")
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    TOKENIZER_LOADED[llm_id] = tokenizer
    return tokenizer
