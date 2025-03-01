import os
from typing import Any

import torch
from dotenv import load_dotenv
from huggingface_hub import login

from meta_evals.models.llms import Llm, LlmId, get_llm
from meta_evals.utils.constants import USE_MPS
from meta_evals.utils.utils import check_for_mps

_loaded_llm_id: LlmId | None = None
_loaded_llm: Llm[Any, Any] | None = None

assert load_dotenv("../.env")
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)


def load_llm_oioo(
    llm_id: LlmId,
    device: torch.device,
    use_half_precision: bool,
) -> Llm[Any, Any]:
    """
    Loads an LLM with a one-in-one-out policy, i.e. only one model is loaded into
    memory at a time.
    """
    global _loaded_llm_id
    global _loaded_llm
    if llm_id != _loaded_llm_id:
        if _loaded_llm is not None:
            _loaded_llm.model = _loaded_llm.model.cpu()
            del _loaded_llm
            if check_for_mps() and USE_MPS:
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()
            print(f"Unloaded LLM {_loaded_llm_id}, loading LLM {llm_id}")
        else:
            print(f"Loading LLM {llm_id}")
        _loaded_llm = get_llm(
            llm_id, device, use_half_precision=use_half_precision
        )
        _loaded_llm_id = llm_id
    assert _loaded_llm is not None
    return _loaded_llm
