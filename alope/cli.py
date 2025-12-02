"""
Simple CLI entrypoints for ALOPE.
"""

import argparse
import json
from dataclasses import asdict

from .config import ExperimentConfig
from .pipeline import run_inference, run_training


def _parse_lang_files(pairs):
    mapping = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--train-file expects lang-pair=path, got {p}")
        key, path = p.split("=", 1)
        mapping[key] = path
    return mapping


def _parse_layer_range(text: str):
    if ":" not in text:
        raise ValueError("--layer-range must be START:END (e.g., 17:25)")
    start, end = text.split(":")
    return int(start), int(end)


def build_parser():
    parser = argparse.ArgumentParser(description="ALOPE QE framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a QE model")
    train_p.add_argument("--model", default=None, help="Model name or path")
    train_p.add_argument("--train-file", action="append", default=[], help="lang-pair=path")
    train_p.add_argument("--val-file", action="append", default=[], help="lang-pair=path")
    train_p.add_argument("--strategy", choices=["single", "multi", "dynamic"], default="multi")
    train_p.add_argument("--layer-range", default="17:25", help="Start:End for layer taps (negative indexing)")
    train_p.add_argument("--output-dir", default="./alope_outputs", help="Where to save checkpoints")
    train_p.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    train_p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_p.add_argument("--epochs", type=float, default=1.0, help="Number of epochs")
    train_p.add_argument("--no-lora", action="store_true", help="Disable LoRA and full finetune instead")
    train_p.add_argument("--aggregation", choices=["average", "learned_softmax"], default="average")
    train_p.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code on backbones")
    train_p.add_argument("--attn-impl", default=None, help="Attention implementation override (sdpa/flash_attention_2)")
    train_p.add_argument("--hf-dataset", default=None, help="HF dataset identifier (e.g., surrey-nlp/Low-resource-QE-DA-dataset)")
    train_p.add_argument("--hf-config", default=None, help="HF dataset config/name (e.g., engu, enhi, multilingual)")
    train_p.add_argument("--hf-train-split", default="train", help="HF train split name")
    train_p.add_argument("--hf-val-split", default="validation", help="HF validation split name")
    train_p.add_argument("--hf-test-split", default="test", help="HF test split name")
    train_p.add_argument("--hf-lang-pair", action="append", default=[], help="HF language pair to keep (repeatable, e.g., en-ta)")
    train_p.add_argument("--hf-original-field", default="original", help="HF column for source text")
    train_p.add_argument("--hf-translation-field", default="translation", help="HF column for target text")
    train_p.add_argument("--hf-score-field", default="mean", help="HF column for QE score")
    train_p.add_argument("--hf-langpair-field", default="lang_pair", help="HF column for language pair")

    infer_p = subparsers.add_parser("infer", help="Run inference with a trained model")
    infer_p.add_argument("--model", required=True, help="Base model used during training")
    infer_p.add_argument("--trained-model", required=True, help="Path to trained checkpoint (folder)")
    infer_p.add_argument("--test-file", help="Path to TSV test file (omit to use HF test split)")
    infer_p.add_argument("--hf-dataset", default=None, help="HF dataset identifier")
    infer_p.add_argument("--hf-config", default=None, help="HF dataset config/name")
    infer_p.add_argument("--hf-test-split", default="test", help="HF test split name")
    infer_p.add_argument("--hf-lang-pair", action="append", default=[], help="HF language pair to keep")
    infer_p.add_argument("--hf-original-field", default="original", help="HF column for source text")
    infer_p.add_argument("--hf-translation-field", default="translation", help="HF column for target text")
    infer_p.add_argument("--hf-score-field", default="mean", help="HF column for QE score")
    infer_p.add_argument("--hf-langpair-field", default="lang_pair", help="HF column for language pair")
    infer_p.add_argument("--layer-range", default="17:25", help="Needed when strategy uses layer taps")
    infer_p.add_argument("--strategy", choices=["single", "multi", "dynamic"], default="multi")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    exp = ExperimentConfig()
    exp.heads.strategy = getattr(args, "strategy", exp.heads.strategy)
    exp.heads.layer_range = _parse_layer_range(getattr(args, "layer_range", "17:25"))
    exp.heads.aggregation = getattr(args, "aggregation", exp.heads.aggregation)

    if getattr(args, "model", None):
        exp.model.model_name_or_path = args.model
    if getattr(args, "trust_remote_code", False):
        exp.model.trust_remote_code = True
    if getattr(args, "attn_impl", None):
        exp.model.attn_implementation = args.attn_impl

    if getattr(args, "no_lora", False):
        exp.lora.enabled = False
    exp.training.output_dir = getattr(args, "output_dir", exp.training.output_dir)
    exp.training.per_device_train_batch_size = getattr(args, "batch_size", exp.training.per_device_train_batch_size)
    exp.training.per_device_eval_batch_size = getattr(args, "batch_size", exp.training.per_device_eval_batch_size)
    exp.training.learning_rate = getattr(args, "lr", exp.training.learning_rate)
    exp.training.num_train_epochs = getattr(args, "epochs", exp.training.num_train_epochs)

    # HF dataset wiring
    if getattr(args, "hf_dataset", None):
        exp.data.hf.dataset_name = args.hf_dataset
        exp.data.hf.dataset_config = getattr(args, "hf_config", None)
        exp.data.hf.train_split = getattr(args, "hf_train_split", exp.data.hf.train_split)
        exp.data.hf.validation_split = getattr(args, "hf_val_split", exp.data.hf.validation_split)
        exp.data.hf.test_split = getattr(args, "hf_test_split", exp.data.hf.test_split)
        exp.data.hf.lang_pairs = getattr(args, "hf_lang_pair", []) or []
        exp.data.hf.original_field = getattr(args, "hf_original_field", exp.data.hf.original_field)
        exp.data.hf.translation_field = getattr(args, "hf_translation_field", exp.data.hf.translation_field)
        exp.data.hf.score_field = getattr(args, "hf_score_field", exp.data.hf.score_field)
        exp.data.hf.lang_pair_field = getattr(args, "hf_langpair_field", exp.data.hf.lang_pair_field)

    if args.command == "train":
        if args.train_file:
            exp.data.train_files = _parse_lang_files(args.train_file)
        if args.val_file:
            exp.data.validation_files = _parse_lang_files(args.val_file)
        trainer, model = run_training(exp)
        print(json.dumps(asdict(exp), indent=2))
    elif args.command == "infer":
        exp.model.model_name_or_path = args.model
        results = run_inference(exp, args.test_file, args.trained_model)
        for idx, truth, pred in results[:5]:
            print(f"{idx}: truth={truth:.3f} pred={pred:.3f}")


if __name__ == "__main__":
    main()
