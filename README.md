# ALOPE

ALOPE is an adaptive layer-optimization framework that enhances quality estimation (QE) for machine translation using large language models (LLMs). It restructures Transformer representations through layer-wise adaptation and integrates low-rank adapters (LoRA) with regression heads, enabling improved regression-based prediction, especially for low-resource languages. ALOPE also introduces dynamic weighting and multi-head regression strategies, adaptively combining information from multiple Transformer layers. The framework is designed to be easily integrated into existing LLMs, enabling robust reference-less quality estimation.

## Framework (UNDER CONSTRUCTION)

The `alope/` package provides configuration dataclasses, head builders, a backbone registry, data loading utilities, and a CLI (`python -m alope.cli`). Defaults aim for compatibility with common decoder families (Llama, Mistral/Mixtral, Gemma, Qwen/Qwen2, Unsloth Llama variants).

Quick starts:

```bash
# Train with per-layer heads averaged (defaults to layers -17..-24)
python -m alope.cli train \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --train-file English-Tamil=data/train.en-ta.tsv \
  --train-file English-Hindi=data/train.en-hi.tsv

# Train with learned layer weighting
python -m alope.cli train \
  --strategy dynamic --aggregation learned_softmax \
  --layer-range 17:25 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --train-file English-Gujarati=data/train.en-gu.tsv

# Inference on a trained checkpoint
python -m alope.cli infer \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --trained-model ./alope_outputs \
  --test-file data/test.engu.tsv
```

The CLI exposes common knobs (model name, strategy, layer range, LoRA toggle, batch size, learning rate, epochs). You can also import `ExperimentConfig` from `alope/config.py` to script custom runs.

### Using Hugging Face datasets (e.g., surrey-nlp/Low-resource-QE-DA-dataset)

Single language-pair config:
```bash
python -m alope.cli train \
  --hf-dataset surrey-nlp/Low-resource-QE-DA-dataset \
  --hf-config engu \
  --hf-lang-pair en-gu \
  --model meta-llama/Llama-3.2-3B-Instruct
```

Multilingual config with filtering and test from HF:
```bash
python -m alope.cli train \
  --hf-dataset surrey-nlp/Low-resource-QE-DA-dataset \
  --hf-config multilingual \
  --hf-lang-pair en-ta --hf-lang-pair en-hi \
  --model mistralai/Mistral-7B-Instruct-v0.3

[For Local Adapter]

python -m alope.cli infer \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --trained-model ./alope_outputs/<model_folder> \
  --hf-dataset surrey-nlp/Low-resource-QE-DA-dataset \
  --hf-config multilingual \
  --hf-lang-pair en-ta \
  --hf-test-split test_enta

[Trained Adapter from HF]

python -m alope.cli infer \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --trained-model surrey-nlp/ALOPE_llama-3.2-3B_TL_-7 \
  --test-file data/test.engu.tsv \
  --strategy single

[If you want to read test from HF instead of a local file (same base/checkpoint)]

python -m alope.cli infer \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --trained-model surrey-nlp/ALOPE_llama-3.2-3B_TL_-7 \
  --strategy single \
  --hf-dataset surrey-nlp/Low-resource-QE-DA-dataset \
  --hf-config engu \
  --hf-lang-pair en-gu \
  --hf-test-split test

```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ArchSid/ALOPE.git
cd ALOPE
```

### 2.Create a new Conda virtual environment
```bash
conda create -n alope python=3.10
conda activate alope
pip install -r requirements.txt
```

Our fine-tuned models adapters with ALOPE framework can be found in the HuggingFace repository:

[https://huggingface.co/surrey-nlp/collections/](https://huggingface.co/surrey-nlp/collections)
