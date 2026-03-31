# Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios

[![arXiv](https://img.shields.io/badge/arXiv-2603.07372-b31b1b.svg)](https://arxiv.org/abs/2603.07372)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow)](https://huggingface.co/collections/surrey-nlp/domain-specific-indic-qe-datasets)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Published at**: [LoResLM @ EACL 2026](https://aclanthology.org/2026.loreslm-1.55/)

## Overview

This repository contains code, data, and experiments for **domain-specific quality estimation (QE) of machine translation** in low-resource scenarios. The work focuses on **English→Indic language translation** across four distinct domains (Healthcare, Legal, Tourism, General) and five languages (Hindi, Marathi, Tamil, Telugu, Gujarati).

We present two complementary approaches:

1. **Prompt-Only LLM Evaluation**: Zero-shot, few-shot, and few-shot-with-guidelines prompting strategies for large language models.
2. **ALOPE** (Adaptive Layer OPtimization for Translation Quality Estimation): A parameter-efficient fine-tuning framework combining LoRA and LoRMA adapters applied to intermediate Transformer layers.

### Key Findings

- **Intermediate Transformer layers (L-9, L-11) consistently outperform the final layer** across models and domains, providing superior feature representations for QE.
- **Closed-weight models achieve strong QE performance even in zero-shot settings**, making them viable for rapid deployment when API access is available.
- **ALOPE with LoRA consistently outperforms prompt-only approaches** for open-weight models, offering substantial improvements in constrained environments.
- **Deployment guidance**: Use closed-weight models with prompting when API access is viable; adopt ALOPE with LoRA for open-weight models in resource-constrained deployments.

---

## Dataset: Indic-Domain-QE

We introduce **Indic-Domain-QE**, a multi-domain, multi-language dataset for MT quality estimation in low-resource settings.

**HuggingFace Collections**: [surrey-nlp/domain-specific-indic-qe-datasets](https://huggingface.co/collections/surrey-nlp/domain-specific-indic-qe-datasets)

### Dataset Statistics

| Domain | Languages | Train | Test |
|--------|-----------|-------|------|
| **Healthcare** | Hindi, Marathi, Tamil, Gujarati | 13,280 | 1,660 |
| **Legal** | Gujarati, Tamil, Telugu | 6,160 | 770 |
| **Tourism** | Hindi, Marathi, Telugu | 13,840 | 1,730 |
| **General** | Hindi, Marathi, Tamil, Telugu, Gujarati | 18,880 | 2,360 |
| **Total** | **5 languages** | **52,160** | **6,520** |

- **DA Scores**: Continuous scale (0–100), averaged from ≥3 independent annotators per sentence pair.
- **Source**: Machine-translated sentences with human quality annotations.

---

### 1. Prompt-Only QE Evaluation

We perform zero-shot, few-shot, and few-shot-with-guidelines evaluations using large language models.

---

### 2. ALOPE Training with LoRA

We also fine-tune open-weight models using LoRA adapters on intermediate Transformer layers.

---

### 3. ALOPE Training with LoRMA

Fine-tune with LoRMA (Low-Rank Multiplicative Adaptation) — unlike LoRA which uses additive low-rank updates, LoRMA adapts models by multiplicatively modulating existing weights, yielding smoother layer-wise behaviour.

---

### 4. Evaluation

We used Spearman correlation, Pearson correlation, and MAE on test sets.

---

## Models Used

### Closed-Weight Models (API-based)
- **Gemini-1.5-Pro** (Google): Strong zero-shot performance, API-based
- **Gemini-2.5-Pro** (Google): Latest variant with improved reasoning

### Open-Weight Models (Local Inference)
- **LLaMA-3.2-3B Instruct**: Lightweight, efficient for edge deployment
- **LLaMA-3.1-8B Instruct**: Balanced performance and size
- **Qwen3-14B**: Multilingual capabilities
- **Gemma-3-27B**: Large-scale open alternative

### ALOPE Fine-tuning Backbone
- **LLaMA-3.2-3B Instruct** with 4-bit QLoRA quantization

---

## Results Summary

### ALOPE vs. Prompt-Only Evaluation
**Metric: Spearman Correlation (ρ) | Model: LLaMA-3.2-3B Instruct (4-bit QLoRA)**

| Domain | Zero-Shot | Best Prompt | ALOPE (LoRA) | ALOPE (LoRMA) |
|--------|-----------|------------|-------------|---------------|
| General | −0.053 | 0.265 | **0.404** | 0.360 |
| Healthcare | 0.073 | 0.285 | **0.415** | 0.355 |
| Legal | −0.205 | 0.118 | **0.431** | 0.265 |
| Tourism | 0.398 | 0.398 | **0.413** | 0.408 |

**Key Observations:**
- ALOPE (LoRA) achieves consistent improvements across all domains
- Intermediate layers (−9, −11) provide superior contextual information
- LoRA maximises ranking accuracy; LoRMA offers robustness when precise layer selection is constrained
- Zero-shot performance varies significantly by domain; ALOPE mitigates this variance

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{gurav2026domain,
  title={Domain-Specific Quality Estimation for Machine Translation in Low-Resource Scenarios},
  author={Gurav, Namrata Patil and Ranu, Akashdeep and Sindhujan, Archchana and Kanojia, Diptesh},
  booktitle={Proceedings of the Second Workshop on Language Models for Low-Resource Languages (LoResLM)},
  pages={XX--XX},
  year={2026},
  month={April},
  address={Valencia, Spain},
  organization={Association for Computational Linguistics},
  url={https://arxiv.org/abs/2603.07372}
}
```

---

## Related Work

This work builds on and extends:

- **ALOPE Framework**: Sindhujan et al. (2025c). "ALOPE: Adaptive Layer Optimization for Parameter-Efficient Translation Quality Estimation." arXiv:2508.07484. [Link](https://arxiv.org/abs/2508.07484), ACL Anthology [Link](https://aclanthology.org/2026.loreslm-1.55/)
- **Data from**: [SurreyNLP-AI](https://github.com/surrey-nlp) — Existing research on low-resource and Indic NLP

---

## Acknowledgements

We thank:
- **University of Surrey** for computational resources
- **School of Computer Science and Electronic Engineering, and Institute for People-Centred AI** for research support
- **EACL 2026 LoResLM Workshop** organizers for the opportunity to present this work

---

## Contact

For questions, issues, or collaboration inquiries:

- **Namrata Patil Gurav**: np00996@surrey.ac.uk
- **Akashdeep Ranu**: ar02258@surrey.ac.uk
- **Archchana Sindhujan**: a.sindhujan@surrey.ac.uk
- **Diptesh Kanojia** (Principal Investigator): d.kanojia@surrey.ac.uk

**Lab**: SurreyNLP-AI, Institute for People-Centred AI, University of Surrey, UK

---
