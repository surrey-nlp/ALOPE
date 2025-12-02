"""
Model assembly utilities using transformer_heads.
"""

from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from transformer_heads import create_headed_qlora, load_headed
from transformer_heads.util.helpers import get_model_params

from .backbones import infer_family_from_name, register_default_backbones
from .config import ExperimentConfig
from .heads import build_dynamic_weight_heads, build_multi_head, build_single_head


class WeightedEmbeddingCombiner(nn.Module):
    """Learns a softmax distribution over layer embeddings."""

    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.hidden_size = hidden_size

    def forward(self, layer_embeddings: List[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(layer_embeddings, dim=0)  # [L, B, T, H]
        weighted = torch.einsum("l,lbtc->btc", weights, stacked)
        return weighted


def _make_quantization_config(qconf) -> BitsAndBytesConfig:
    if not qconf.load_in_4bit and not qconf.load_in_8bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=qconf.load_in_4bit,
        load_in_8bit=qconf.load_in_8bit,
        llm_int8_threshold=qconf.llm_int8_threshold,
        llm_int8_has_fp16_weight=qconf.llm_int8_has_fp16_weight,
        bnb_4bit_compute_dtype=getattr(torch, qconf.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=qconf.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=qconf.bnb_4bit_quant_type,
    )


def _make_lora_config(lconf: ExperimentConfig) -> LoraConfig:
    return LoraConfig(
        r=lconf.lora.r,
        lora_alpha=lconf.lora.alpha,
        target_modules=lconf.lora.target_modules,
        lora_dropout=lconf.lora.dropout,
        bias=lconf.lora.bias,
        task_type=lconf.lora.task_type,
    )


def _build_heads(hidden_size: int, exp: ExperimentConfig):
    strategy = exp.heads.strategy
    if strategy == "single":
        return build_single_head(hidden_size, loss_weight=exp.heads.loss_weight, name=exp.heads.regression_head_name)
    if strategy == "multi":
        return build_multi_head(
            hidden_size,
            exp.heads.layer_range,
            per_layer_loss_weight=exp.heads.per_layer_loss_weight,
            regression_loss_weight=exp.heads.loss_weight,
            regression_head_name=exp.heads.regression_head_name,
            aggregation=exp.heads.aggregation,
        )
    if strategy == "dynamic":
        return build_dynamic_weight_heads(
            hidden_size,
            exp.heads.layer_range,
            regression_loss_weight=exp.heads.loss_weight,
            regression_head_name=exp.heads.regression_head_name,
        )
    raise ValueError(f"Unknown head strategy: {strategy}")


def _attach_dynamic_aggregation(model, exp: ExperimentConfig, embed_head_names: Iterable[str]):
    embed_head_names = list(embed_head_names)
    combiner = WeightedEmbeddingCombiner(num_layers=len(embed_head_names), hidden_size=model.config.hidden_size)
    original_forward = model.forward

    def custom_forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        preds = getattr(outputs, "preds_by_head", None) or outputs.get("preds_by_head")
        if not preds:
            return outputs
        if all(name in preds for name in embed_head_names):
            embeddings = [preds[name] for name in embed_head_names]
            combined = combiner(embeddings)
            preds[exp.heads.regression_head_name] = self.heads[exp.heads.regression_head_name](combined)
        outputs["preds_by_head"] = preds
        return outputs

    model.forward = custom_forward.__get__(model, type(model))
    model.embedding_combiner = combiner


def build_model(exp: ExperimentConfig):
    register_default_backbones()
    model_params = get_model_params(exp.model.model_name_or_path)
    hidden_size = model_params["hidden_size"]
    model_class = model_params["model_class"]

    head_configs = _build_heads(hidden_size, exp)

    quantization_config = _make_quantization_config(exp.quantization)
    lora_config = _make_lora_config(exp)

    if exp.lora.enabled:
        model = create_headed_qlora(
            base_model_class=model_class,
            model_name=exp.model.model_name_or_path,
            quantization_config=quantization_config,
            lora_config=lora_config,
            head_configs=head_configs,
            fully_trained_heads=True,
            device_map={"": "auto"},
            gradient_checkpointing=exp.model.gradient_checkpointing,
            trust_remote_code=exp.model.trust_remote_code,
            attn_implementation=exp.model.attn_implementation,
        )
    else:
        model = load_headed(
            base_model_class=model_class,
            model_name=exp.model.model_name_or_path,
            head_configs=head_configs,
            device_map={"": "auto"},
            quantization_config=quantization_config,
            freeze_base_model=False,
            trust_remote_code=exp.model.trust_remote_code,
            attn_implementation=exp.model.attn_implementation,
        )

    if exp.heads.strategy == "dynamic":
        start, end = exp.heads.layer_range
        embed_names = [f"layer_{i}_embed" for i in range(start, end)]
        _attach_dynamic_aggregation(model, exp, embed_names)

    return model, head_configs, hidden_size

