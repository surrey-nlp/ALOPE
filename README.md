# ALOPE: Adaptive Layer Optimization for Translation Quality Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2508.07484-b31b1b.svg)](https://arxiv.org/abs/2508.07484)
[![COLM 2025](https://img.shields.io/badge/COLM-2025-blue)](https://colmweb.org/2025/AcceptedPapers.html)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/collections/surrey-nlp/alope-models-68dd1d660e3249cb6b3f4977)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Published at**: [COLM 2025](https://colmweb.org/2025/AcceptedPapers.html) &bull; [arXiv](https://arxiv.org/abs/2508.07484)

## Overview

**ALOPE** is an adaptive layer-optimization framework that enhances Quality Estimation (QE) for machine translation using large language models. QE evaluates the quality of a source–target translation pair **without reference translations**, making it essential for real-world MT deployment.

ALOPE restructures Transformer representations through **layer-wise adaptation**, integrating **LoRA (Low-Rank Adaptation)** with regression task heads on selected intermediate Transformer layers. The key insight is that **intermediate Transformer layers provide superior cross-lingual representations** compared to the commonly used final layer.

### Key Findings

- **Intermediate layers outperform the final layer**: Transformer Layer −7 (TL-7) delivers the best overall QE performance across models and language pairs, with TL-11 as a strong alternative.
- **Three complementary strategies**: Single-layer regression, dynamic weighting, and multi-head regression offer flexibility depending on deployment constraints.
- **Consistent improvements over baselines**: ALOPE achieves results comparable to established QE frameworks while being parameter-efficient through 4-bit QLoRA quantization.
- **Strong cross-lingual transfer**: Effective across 8 language pairs spanning both English→Indic and Indic→English directions.

---

## Three ALOPE Strategies

### 1. Single-Layer Regression
Extracts hidden representations from a **single intermediate Transformer layer** and trains a regression head on top for DA (Direct Assessment) score prediction. This is the simplest and most interpretable approach — used to identify which layers carry the most QE-relevant information.

### 2. Dynamic Weighting
Adaptively **combines representations from multiple Transformer layers** using learned weights. Rather than committing to a single layer, this strategy lets the model discover the optimal blend of layer-wise features for each input, capturing both syntactic (lower layers) and semantic (higher layers) information.

### 3. Multi-Head Regression
Attaches **independent regression heads to multiple Transformer layers** and aggregates their losses during training. Each head specializes on the representation quality of its assigned layer, and the combined signal provides a more robust QE prediction.

---

## Language Pairs

ALOPE is evaluated on **8 language pairs** across two directions:

| Direction | Language Pairs |
|-----------|---------------|
| **English → Indic** | En→Gu, En→Hi, En→Mr, En→Ta, En→Te |
| **Indic → English** | Ne→En, Et→En, Si→En |

---

## Models

### Fine-tuning Backbones (with 4-bit QLoRA)

| Model | Parameters | Source |
|-------|-----------|--------|
| **LLaMA-2-7B** | 7B | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |
| **LLaMA-3.1-8B Instruct** | 8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| **LLaMA-3.2-3B Instruct** | 3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| **Aya-Expanse-8B** | 8B | [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) |

### Transformer Layers Explored

Layers **{−1, −7, −11, −16, −20, −24}** are evaluated across all models. **TL-7 (Layer −7)** emerges as the best-performing layer overall, with correlation scores declining beyond mid-Transformer depth (TL-16) for most models.

---

## Fine-Tuned Models on HuggingFace

All ALOPE fine-tuned models are publicly available:

| Collection | Link |
|-----------|------|
| **All ALOPE Models** | [surrey-nlp/alope-models](https://huggingface.co/collections/surrey-nlp/alope-models-68dd1d660e3249cb6b3f4977) |
| LLaMA-2-7B variants | [surrey-nlp/alope-llama-2-7b-models](https://huggingface.co/collections/surrey-nlp/alope-llama-2-7b-models-68b024fe6ded8dc5958aef71) |
| LLaMA-3.1-8B variants | [surrey-nlp/alope-llama-31-8b-models](https://huggingface.co/collections/surrey-nlp/alope-llama-31-8b-models-68b0269d90d9ea9a546ebe44) |
| LLaMA-3.2-3B variants | [surrey-nlp/alope-llama-32-3b-models](https://huggingface.co/collections/surrey-nlp/alope-llama-32-3b-models-68b027e4555d5f6418d45055) |
| Aya-Expanse-8B variants | [surrey-nlp/alope-aya-expanse-8b-models](https://huggingface.co/collections/surrey-nlp/alope-aya-expanse-8b-models-68b028a990d9ea9a546efeae) |
| ALOPE-RL Models | [surrey-nlp/alope-rl-models](https://huggingface.co/collections/surrey-nlp/alope-rl-models-696a98a1776f8ff4db5fd881) |

---

## Repository Structure

```
ALOPE/
├── single-layer-regression_train.py       # Strategy 1: Train single-layer regression
├── single-layer-regression_inference.py   # Strategy 1: Inference
├── Dynamic_weighting-train.py             # Strategy 2: Train dynamic weighting
├── Dynamic_weighting-inference.py         # Strategy 2: Inference
├── Multi-head_regression_train.py         # Strategy 3: Train multi-head regression
├── Multi-head_regression_inference.py     # Strategy 3: Inference
├── FT-without_LORA.PY                     # Baseline: Fine-tuning without LoRA
├── requirements.txt                       # Python dependencies
├── Domain-based-QE-with-ALOPE/            # Domain-specific QE extension (LoResLM @ EACL 2026)
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/surrey-nlp/ALOPE.git
cd ALOPE
```

### 2. Create a Conda environment

```bash
conda create -n alope python=3.10
conda activate alope
pip install -r requirements.txt
```

---

## Usage

### Single-Layer Regression

**Training** — fine-tune a LoRA adapter on a specific intermediate layer:
```bash
python single-layer-regression_train.py
```

**Inference** — predict DA scores using the trained adapter:
```bash
python single-layer-regression_inference.py
```

### Dynamic Weighting

**Training** — learn adaptive layer weights across multiple Transformer layers:
```bash
python Dynamic_weighting-train.py
```

**Inference**:
```bash
python Dynamic_weighting-inference.py
```

### Multi-Head Regression

**Training** — train independent regression heads on multiple layers:
```bash
python Multi-head_regression_train.py
```

**Inference**:
```bash
python Multi-head_regression_inference.py
```

### Baseline (without LoRA)

```bash
python FT-without_LORA.PY
```

---

## Results Summary

ALOPE is evaluated using **Spearman's rank correlation (ρ)** between predicted and human DA scores. Key findings across all 8 language pairs and 4 models:

- **TL-7 consistently achieves the highest Spearman correlations** across most model–language pair combinations.
- **Dynamic weighting and multi-head regression** provide robust alternatives when optimal single-layer selection is uncertain.
- **ALOPE achieves results comparable to established QE frameworks** (e.g., CometKiwi) while using significantly fewer trainable parameters through 4-bit QLoRA.
- Correlation scores generally **decrease beyond mid-Transformer depth (TL-16)** for most models.

For full results tables across all models, layers, and language pairs, see the [paper](https://arxiv.org/abs/2508.07484) (Tables 1–2).

---

## Domain-Specific Extension

The `Domain-based-QE-with-ALOPE/` subdirectory extends ALOPE to **domain-specific QE** for English→Indic translation across Healthcare, Legal, Tourism, and General domains. This work introduces **LoRMA (Low-Rank Multiplicative Adaptation)** as a complementary adaptation strategy alongside LoRA.

- **Paper**: [Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios](https://arxiv.org/abs/2603.07372)
- **Venue**: [LoResLM Workshop @ EACL 2026](https://aclanthology.org/2026.loreslm-1.55/)
- **Datasets**: [surrey-nlp/domain-specific-indic-qe-datasets](https://huggingface.co/collections/surrey-nlp/domain-specific-indic-qe-datasets-697baf3091d3f200237ae4fa)

See the [subdirectory README](Domain-based-QE-with-ALOPE/README.md) for full details.

---

## Citation

If you use this code, framework, or models, please cite:

### ALOPE (COLM 2025)

```bibtex
@inproceedings{sindhujan2025alope,
  title={ALOPE: Adaptive Layer Optimization for Translation Quality Estimation using Large Language Models},
  author={Sindhujan, Archchana and Qian, Shenbin and Chan, Chi Chun Matthew and Ora{\c{s}}an, Constantin and Kanojia, Diptesh},
  booktitle={Proceedings of the Conference on Language Modeling (COLM)},
  year={2025},
  url={https://arxiv.org/abs/2508.07484}
}
```

### Domain-Specific QE Extension (LoResLM @ EACL 2026)

```bibtex
@inproceedings{gurav-etal-2026-domain,
    title = "Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios",
    author = "Gurav, Namrata Bhalchandra Patil  and
      Ranu, Akashdeep  and
      Sindhujan, Archchana  and
      Kanojia, Diptesh",
    editor = "Hettiarachchi, Hansi  and
      Ranasinghe, Tharindu  and
      Plum, Alistair  and
      Rayson, Paul  and
      Mitkov, Ruslan  and
      Gaber, Mohamed  and
      Premasiri, Damith  and
      Tan, Fiona Anting  and
      Uyangodage, Lasitha",
    booktitle = "Proceedings of the Second Workshop on Language Models for Low-Resource Languages ({L}o{R}es{LM} 2026)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.loreslm-1.55/",
    doi = "10.18653/v1/2026.loreslm-1.55",
    pages = "630--650",
    ISBN = "979-8-89176-377-7",
    abstract = "Quality Estimation (QE) is essential for assessing machine translation quality in reference-less settings, particularly for domain-specific and low-resource language scenarios. In this paper, we investigate sentence-level QE for English to Indic machine translation across four domains (Healthcare, Legal, Tourism, and General) and five language pairs. We systematically compare zero-shot, few-shot, and guideline-anchored prompting across selected closed-weight and open-weight LLMs. Findings indicate that while closed-weight models achieve strong performance via prompting alone, prompt-only approaches remain fragile for open-weight models, especially in high-risk domains. To address this, we adopt ALOPE, a framework for LLM-based QE which uses Low-Rank Adaptation with regression heads attached to selected intermediate Transformer layers. We also extend ALOPE with the recently proposed Low-Rank Multiplicative Adaptation (LoRMA) for this work. Our results show that intermediate-layer adaptation consistently improves QE performance, with gains in semantically complex domains, indicating a way ahead for robust QE in practical scenarios. We release code and domain-specific QE datasets publicly for further research."
}
```

---

## Acknowledgements

We thank the **Institute for People-Centred AI, University of Surrey** for computational resources and the **School of Computer Science and Electronic Engineering** for research support.

---

## Contact

For questions, issues, or collaboration inquiries:

- **Archchana Sindhujan**: a.sindhujan@surrey.ac.uk
- **Diptesh Kanojia** (PI): d.kanojia@surrey.ac.uk

**Lab**: [SurreyNLP](https://github.com/surrey-nlp), Institute for People-Centred AI, University of Surrey, UK
