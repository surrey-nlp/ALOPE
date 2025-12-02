"""
Data loading and validation for QE datasets.
"""

import csv
from typing import Dict, Iterable, List, Tuple, Optional

import pandas as pd
from datasets import Dataset, concatenate_datasets

from .config import HFDataConfig, DataConfig

from .prompts import render_prompt

REQUIRED_COLUMNS = {"original", "translation", "mean"}


def _validate_columns(df: pd.DataFrame, path: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_pair_dataset(
    lang_pair: str,
    file_path: str,
    sep: str = "\t",
    max_samples: int = None,
) -> Dataset:
    df = pd.read_csv(file_path, sep=sep, quoting=csv.QUOTE_NONE)
    _validate_columns(df, file_path)
    if max_samples:
        df = df.head(max_samples)
    src_lang, tgt_lang = lang_pair.split("-")
    df["source_lang"] = src_lang
    df["target_lang"] = tgt_lang
    return Dataset.from_pandas(df)


def load_train_datasets(
    train_files: Dict[str, str],
    sep: str = "\t",
    max_samples: int = None,
) -> Dataset:
    datasets: List[Dataset] = []
    for lang_pair, path in train_files.items():
        datasets.append(load_pair_dataset(lang_pair, path, sep, max_samples))
    if not datasets:
        raise ValueError("No training files provided.")
    return concatenate_datasets(datasets)


def add_prompts(dataset: Dataset, template: str) -> Dataset:
    def _process(examples):
        sources = examples["original"]
        targets = examples["translation"]
        src_langs = examples["source_lang"]
        tgt_langs = examples["target_lang"]
        prompts = [
            render_prompt(template, s, t, src, tgt)
            for s, t, src, tgt in zip(sources, targets, src_langs, tgt_langs)
        ]
        return {"prompt": prompts}

    return dataset.map(_process, batched=True)


def trim_columns_for_training(dataset: Dataset, head_names: Iterable[str]) -> Dataset:
    to_remove = {
        "original",
        "translation",
        "scores",
        "z_scores",
        "z_mean",
        "source_lang",
        "target_lang",
        "prompt",
        "index",
    }
    to_remove -= set(head_names)
    keep = [c for c in dataset.column_names if c not in to_remove]
    return dataset.remove_columns([c for c in dataset.column_names if c not in keep])


def _parse_lang_pair(code: str) -> Tuple[str, str]:
    if "-" in code:
        src, tgt = code.split("-", 1)
        return src, tgt
    if len(code) == 4:  # e.g., eten -> et-en
        return code[:2], code[2:]
    if len(code) == 5 and code[2] == "_":  # e.g., en_hi
        return code[:2], code[3:]
    raise ValueError(f"Cannot parse language pair from '{code}'")


def _maybe_trim(ds: Dataset, limit: Optional[int]) -> Dataset:
    if limit is None:
        return ds
    return ds.select(range(min(limit, len(ds))))


def load_hf_split(
    hf_conf: HFDataConfig,
    split_name: str,
) -> Optional[Dataset]:
    if not hf_conf.dataset_name:
        return None
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError("datasets package is required for HF loading") from exc

    load_kwargs = {"name": hf_conf.dataset_config} if hf_conf.dataset_config else {}
    try:
        ds_dict = load_dataset(hf_conf.dataset_name, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Failed to load HF dataset {hf_conf.dataset_name} ({hf_conf.dataset_config}): {exc}") from exc

    # Prefer exact split name; fallback to per-lang test split (e.g., test_engu)
    if split_name in ds_dict:
        ds = ds_dict[split_name]
    else:
        # Try per-language suffix
        for lp in hf_conf.lang_pairs:
            candidate = f"{split_name}_{lp.replace('-', '')}"
            if candidate in ds_dict:
                ds = ds_dict[candidate]
                break
        else:
            return None

    lang_pairs = hf_conf.lang_pairs or []
    lp_field = hf_conf.lang_pair_field

    def _select_lang_pairs(example_lp: str) -> bool:
        if not lang_pairs:
            return True
        if example_lp in lang_pairs:
            return True
        # tolerate formats without dash
        for lp in lang_pairs:
            if lp.replace("-", "") == example_lp.replace("-", ""):
                return True
        return False

    # Filter multilingual splits if a lang_pair field exists
    if lp_field in ds.column_names:
        if lang_pairs:
            ds = ds.filter(lambda x: _select_lang_pairs(x[lp_field]))
    else:
        # If no lang_pair field, ensure we have exactly one requested pair
        if len(lang_pairs) > 1:
            raise ValueError("HF split lacks lang_pair field; specify at most one lang pair")
        if len(lang_pairs) == 1:
            ds = ds.add_column(lp_field, [lang_pairs[0]] * len(ds))

    # Map columns
    def _rename(example):
        lp_val = example.get(lp_field, hf_conf.lang_pairs[0] if hf_conf.lang_pairs else "en-en")
        src_lang, tgt_lang = _parse_lang_pair(str(lp_val))
        return {
            "original": example[hf_conf.original_field],
            "translation": example[hf_conf.translation_field],
            "mean": example[hf_conf.score_field],
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "lang_pair": f"{src_lang}-{tgt_lang}",
        }

    ds = ds.map(_rename, remove_columns=[c for c in ds.column_names if c not in {hf_conf.original_field, hf_conf.translation_field, hf_conf.score_field, lp_field}])

    return ds


def combine_train_sources(data_conf: DataConfig) -> Dataset:
    datasets: List[Dataset] = []
    if data_conf.train_files:
        datasets.append(load_train_datasets(data_conf.train_files, sep=data_conf.sep, max_samples=data_conf.max_train_samples))
    hf_ds = load_hf_split(data_conf.hf, data_conf.hf.train_split) if data_conf.hf and data_conf.hf.dataset_name else None
    if hf_ds:
        hf_ds = _maybe_trim(hf_ds, data_conf.hf.max_train_samples)
        datasets.append(hf_ds)
    if not datasets:
        raise ValueError("No training data provided (local or HF).")
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def combine_eval_sources(data_conf: DataConfig) -> Optional[Dataset]:
    datasets: List[Dataset] = []
    if data_conf.validation_files:
        datasets.append(load_train_datasets(data_conf.validation_files, sep=data_conf.sep, max_samples=data_conf.max_eval_samples))
    hf_ds = load_hf_split(data_conf.hf, data_conf.hf.validation_split) if data_conf.hf and data_conf.hf.dataset_name else None
    if hf_ds:
        hf_ds = _maybe_trim(hf_ds, data_conf.hf.max_eval_samples)
        datasets.append(hf_ds)
    if not datasets:
        return None
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def load_hf_test_split(data_conf: DataConfig) -> Optional[Dataset]:
    if not (data_conf.hf and data_conf.hf.dataset_name):
        return None
    ds = load_hf_split(data_conf.hf, data_conf.hf.test_split)
    if ds and data_conf.hf.max_test_samples:
        ds = _maybe_trim(ds, data_conf.hf.max_test_samples)
    return ds
