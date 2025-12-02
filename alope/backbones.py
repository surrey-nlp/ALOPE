"""
Backbone registry helpers.

We extend transformer_heads' model_type_map with popular decoder-only families:
Llama, Mistral/Mixtral, Gemma, Qwen/Qwen2, and Unsloth-quantized Llama.
Imports are guarded so the module works even if a specific family is missing
from the installed transformers wheel.
"""

from typing import Dict, Optional, Tuple

from transformer_heads.constants import model_type_map


def _safe_import(path: str):
    module_name, class_name = path.rsplit(".", 1)
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception:
        return None


DEFAULT_BACKBONES = {
    "meta-llama": "transformers.LlamaForCausalLM",
    "mistral": "transformers.MistralForCausalLM",
    "mixtral": "transformers.MixtralForCausalLM",
    "mistral-nemo": "transformers.MistralForCausalLM",
    "qwen": "transformers.Qwen2ForCausalLM",
    "qwen2": "transformers.Qwen2ForCausalLM",
    "qwen2-moe": "transformers.Qwen2MoeForCausalLM",
    "gemma": "transformers.GemmaForCausalLM",
    "google/gemma": "transformers.GemmaForCausalLM",
    "unsloth": "unsloth.models.Autograd4bitLlamaModel",  # optional
}


def register_default_backbones() -> Dict[str, Tuple[str, object]]:
    """
    Populate transformer_heads.model_type_map with decoder backbones we expect
    to use. Returns the updated map for inspection.
    """
    for key, import_path in DEFAULT_BACKBONES.items():
        cls = _safe_import(import_path)
        if cls:
            model_type_map[key] = ("model", cls)
    return model_type_map


def infer_family_from_name(model_name_or_path: str) -> Optional[str]:
    lowered = model_name_or_path.lower()
    for family in DEFAULT_BACKBONES:
        if family in lowered:
            return family
    return None

