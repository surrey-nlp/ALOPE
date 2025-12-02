"""
High-level training and inference routines.
"""

from typing import Iterable, List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformer_heads.util.helpers import get_model_params

from .collators import build_head_collator
from .config import ExperimentConfig
from .data import (
    add_prompts,
    trim_columns_for_training,
    combine_train_sources,
    combine_eval_sources,
    load_hf_test_split,
)
from .modeling import build_model


def head_names_with_loss(head_configs) -> List[str]:
    return [h.name for h in head_configs if h.loss_fct is not None and h.loss_weight != 0.0]


def build_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_dataset(dataset: Dataset, tokenizer, head_configs, exp: ExperimentConfig) -> Dataset:
    label_heads = head_names_with_loss(head_configs)

    def _process(examples):
        prompts = examples["prompt"]
        out = tokenizer(prompts, padding=False, truncation=True)
        for head_name in label_heads:
            out[head_name] = examples["mean"]
        return out

    tokenized = dataset.map(_process, batched=True, remove_columns=[])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"] + label_heads)
    tokenized = trim_columns_for_training(tokenized, label_heads)
    return tokenized


def prepare_train_data(exp: ExperimentConfig, tokenizer) -> Dataset:
    raw = combine_train_sources(exp.data)
    with_prompts = add_prompts(raw, exp.data.prompt_template)
    return with_prompts


def prepare_eval_data(exp: ExperimentConfig, tokenizer) -> Dataset:
    raw = combine_eval_sources(exp.data)
    if raw is None:
        return None
    with_prompts = add_prompts(raw, exp.data.prompt_template)
    return with_prompts


def run_training(exp: ExperimentConfig):
    model, head_configs, hidden_size = build_model(exp)
    tokenizer = build_tokenizer(exp.model.model_name_or_path)

    train_dataset = prepare_train_data(exp, tokenizer)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, head_configs, exp)

    eval_dataset = prepare_eval_data(exp, tokenizer)
    if eval_dataset:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, head_configs, exp)

    label_heads = head_names_with_loss(head_configs)
    collator = build_head_collator(tokenizer, label_heads)

    training_args = TrainingArguments(
        output_dir=exp.training.output_dir,
        learning_rate=exp.training.learning_rate,
        weight_decay=exp.training.weight_decay,
        warmup_ratio=exp.training.warmup_ratio,
        num_train_epochs=exp.training.num_train_epochs,
        logging_steps=exp.training.logging_steps,
        do_eval=exp.training.eval_strategy != "no",
        evaluation_strategy=exp.training.eval_strategy,
        eval_steps=exp.training.eval_steps,
        save_steps=exp.training.save_steps,
        save_total_limit=exp.training.save_total_limit,
        per_device_train_batch_size=exp.training.per_device_train_batch_size,
        per_device_eval_batch_size=exp.training.per_device_eval_batch_size,
        gradient_accumulation_steps=exp.training.gradient_accumulation_steps,
        max_grad_norm=exp.training.max_grad_norm,
        lr_scheduler_type=exp.training.lr_scheduler_type,
        optim=exp.training.optim,
        gradient_checkpointing=exp.model.gradient_checkpointing,
        fp16=exp.training.fp16,
        bf16=exp.training.bf16,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(exp.training.output_dir)
    tokenizer.save_pretrained(exp.training.output_dir)
    return trainer, model


def run_inference(exp: ExperimentConfig, test_path: Optional[str], trained_model_path: str):
    # Lightweight inference routine matching the training pipeline.
    from transformer_heads import load_lora_with_heads
    from transformer_heads.output import HeadedModelOutput
    from transformers import BitsAndBytesConfig
    from transformer_heads.config import HeadConfig
    import json
    from datasets import Dataset
    import pandas as pd
    from torch.utils.data import DataLoader

    tokenizer = build_tokenizer(exp.model.model_name_or_path)

    head_configs_path = f"{trained_model_path.rstrip('/')}/head_configs.json"
    with open(head_configs_path, "r") as f:
        head_configs_data = json.load(f)
    head_configs = [HeadConfig(**cfg) for cfg in head_configs_data.values()]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=exp.quantization.load_in_4bit,
        load_in_8bit=exp.quantization.load_in_8bit,
        llm_int8_threshold=exp.quantization.llm_int8_threshold,
        llm_int8_has_fp16_weight=exp.quantization.llm_int8_has_fp16_weight,
        bnb_4bit_compute_dtype=getattr(torch, exp.quantization.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=exp.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=exp.quantization.bnb_4bit_quant_type,
    )

    model_params = get_model_params(exp.model.model_name_or_path)
    model_class = model_params["model_class"]

    model = load_lora_with_heads(
        model_class,
        trained_model_path,
        quantization_config,
        device_map={"": "auto"},
    )
    model_device = next(model.parameters()).device

    if test_path:
        language_pair = test_path.split("/")[-1].split(".")[1]
        src = language_pair[:2]
        tgt = language_pair[2:]
        raw = Dataset.from_pandas(pd.read_csv(test_path, sep=exp.data.sep))
        raw = raw.add_column("source_lang", [src] * len(raw))
        raw = raw.add_column("target_lang", [tgt] * len(raw))
    else:
        raw = load_hf_test_split(exp.data)
        if raw is None:
            raise ValueError("No test data provided (test_file or HF test split).")
    raw = add_prompts(raw, exp.data.prompt_template)
    tokenized = tokenize_dataset(raw, tokenizer, head_configs, exp)

    label_heads = head_names_with_loss(head_configs)
    collator = build_head_collator(tokenizer, label_heads)
    loader = DataLoader(tokenized, batch_size=1, collate_fn=collator)

    results = []
    for idx, batch in enumerate(loader):
        batch = {k: v.to(model_device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        preds = output.preds_by_head
        if exp.heads.strategy == "multi":
            per_layer = [preds[h][0, -1, 0].item() for h in preds if h.startswith("regression_head_")]
            if per_layer:
                value = sum(per_layer) / len(per_layer)
            else:
                value = preds[exp.heads.regression_head_name][0, -1, 0].item()
        else:
            value = preds[exp.heads.regression_head_name][0, -1, 0].item()
        truth = batch[label_heads[0]].item()
        results.append((idx, truth, value))
    return results
