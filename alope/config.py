"""
Typed configuration objects for the ALOPE framework.

These dataclasses capture the knobs we expose via CLI/YAML while keeping
robust defaults that are mathematically sane for QE regression.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Locations and schema hints for QE data."""

    train_files: Dict[str, str] = field(
        default_factory=lambda: {
            # language_pair: path_to_tsv
            # Example: "English-Tamil": "data/train.en-ta.df.short.tsv",
        }
    )
    validation_files: Dict[str, str] = field(default_factory=dict)
    test_files: List[str] = field(default_factory=list)
    sep: str = "\t"
    prompt_template: str = (
        'Score the following translation from {src_lang} to {tgt_lang} on a '
        'continuous scale from 0 to 100, where 0 means "no meaning preserved" '
        'and 100 means "perfect meaning and grammar". {src_lang} source: '
        '"{source}" {tgt_lang} translation: "{target}" Score: '
    )
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    hf: "HFDataConfig" = field(default_factory=lambda: HFDataConfig())


@dataclass
class HFDataConfig:
    """
    Configuration for loading QE data from Hugging Face datasets.

    dataset_name: HF identifier (e.g., "surrey-nlp/Low-resource-QE-DA-dataset")
    dataset_config: HF config/name (e.g., "engu" or "multilingual")
    split names default to train/validation/test but can be overridden.
    lang_pairs: list of lang-pair codes to keep (e.g., ["en-ta", "en-hi"]).
    column mapping lets us adapt arbitrary schemas.
    """

    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    lang_pairs: List[str] = field(default_factory=list)
    original_field: str = "original"
    translation_field: str = "translation"
    score_field: str = "mean"
    lang_pair_field: str = "lang_pair"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class QuantizationConfig:
    """bnb quantization defaults chosen for stability on recent decoder LLMs."""

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_compute_dtype: str = "float32"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


@dataclass
class LoRAConfig:
    """LoRA defaults that trade off quality and stability for QE regression."""

    r: int = 32
    alpha: int = 16
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    task_type: str = "CAUSAL_LM"
    enabled: bool = True  # allow full finetune if False


@dataclass
class HeadStrategyConfig:
    """
    Head definitions and aggregation options.

    strategy: "single" (last layer), "multi" (per-layer heads), or
              "dynamic" (per-layer heads + learned weighting).
    """

    strategy: str = "multi"
    layer_range: Tuple[int, int] = (17, 25)  # inclusive start, exclusive end
    regression_head_name: str = "mean_regression"
    loss_weight: float = 2e-3
    per_layer_loss_weight: float = 2e-4
    aggregation: str = "average"  # "average" or "learned_softmax"


@dataclass
class ModelConfig:
    """Backbone-specific configuration."""

    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    attn_implementation: Optional[str] = None  # e.g., "sdpa" or "flash_attention_2"
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    use_lm_head_loss: bool = False  # keep QE regression losses only by default


@dataclass
class TrainingConfig:
    """Training hyperparameters with conservative defaults."""

    output_dir: str = "./alope_outputs"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    num_train_epochs: float = 1.0
    logging_steps: int = 20
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 2
    eval_strategy: str = "no"  # "no", "steps", or "epoch"
    fp16: bool = True
    bf16: bool = False


@dataclass
class ExperimentConfig:
    """Top-level config bundling all pieces."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    heads: HeadStrategyConfig = field(default_factory=HeadStrategyConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
