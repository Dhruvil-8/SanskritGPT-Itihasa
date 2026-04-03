---
title: SanskritGPT-Itihasa
emoji: 🕉
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
python_version: 3.11
pinned: false
---

# SanskritGPT-Itihasa: Computational Research and Generative Suite

SanskritGPT-Itihasa is a transformer-based framework designed for the computational analysis and generative modeling of the Mahabharata and Ramayana. This project represents an AI-assisted model training and coding experiment focused on capturing the structural, metrical, and stylistic essence of the Sanskrit Itihasa corpus. The model was trained from scratch using original Devanagari texts of the Mahabharata and Ramayana on a Google Colab T4 GPU.

## Project Scope and Objectives

The primary objective of this project is to explore the capacity of modern language models to learn the complex linguistic regulations of Classical Sanskrit. This is a computational linguistic experiment and should be treated as a research tool rather than an authoritative source of scripture.

## Current Model Capabilities

The current iteration of the model includes the following capabilities:

- **High-Precision Training**: Optimized for high metrical adherence to the Anushtubh meter. The model was trained from scratch using original Devanagari texts of the Mahabharata and Ramayana on a Google Colab T4 GPU infrastructure.
- **Style-Conditioned Generation**: The model utilizes control tokens (`<MBH>` and `<RAM>`) to generate text specifically in the stylistic register of either the Mahabharata or the Ramayana.
- **Linguistic Precision**: Trained with a 512-token context window for 30 epochs, the model captures long-range dependencies and stylistic tropes unique to the epics.
- **Automated Filtering**: Includes a post-processing layer to strip metadata tags and sub-word artifacts, ensuring professional-grade output.

## Model Specifications

| Attribute | Value |
| :--- | :--- |
| **Architecture** | GPT-2 (Decoder-only Transformer) |
| **Parameters** | ~42 Million |
| **Layers** | 8 |
| **Attention Heads** | 8 |
| **Embedding Dimension** | 512 |
| **Context Window** | 512 Tokens |
| **Vocabulary Size** | 32,000 (BPE) |
| **Weight File Size** | ~160 MB (Safetensors) |
| **Training Hardware** | Google Colab T4 GPU |
| **Training Epochs** | 30 |

## Repository Contents

This repository includes all components required to reproduce or extend the project:

- `data/raw/` — Original Devanagari source texts (18 Parvas of the Mahabharata and 7 Kandas of the Ramayana)
- `data/processed/` — Consolidated and cleaned training corpus
- `notebooks/epic_model_training.ipynb` — Complete training workbook with hyperparameter configurations
- `notebooks/Sanskrit_Epic_Research_Suite.ipynb` — Unified structural and linguistic analysis suite
- `evaluation_report.md` — Quantitative metrics (Perplexity) and qualitative generation samples
- `app.py` — Gradio deployment interface

## How to Run the App

```bash
pip install -r requirements.txt
python app.py
```

## Limitations

Users should be aware of the following limitations:

- **Statistical Modeling**: The model operates on statistical pattern matching and does not possess an understanding of traditional Sanskrit semantics or philosophical context.
- **Training Bias**: Given the finite size of the epic corpus, the model may occasionally exhibit memorization of specific verses rather than purely original generation.
- **Contextual Boundaries**: Generation is limited by a static 512-token context window.
- **Research Status**: This is an experimental model and may occasionally produce synthetic Sanskrit that deviates from classical grammar.
- **Not Authoritative Scripture**: Generated text is not authentic Vedic or epic scripture and should not be used for ritual, recitation, or scholarly interpretation of tradition.

## Project Structure

```text
Epic/
├── data/
│   ├── raw/                      <- Original Devanagari epic sources (MBH and RAM)
│   └── processed/                <- Final merged and cleaned training corpus
├── model/
│   └── Epic/                     <- Trained model weights and tokenizer config
├── notebooks/
│   ├── epic_model_training.ipynb <- Primary training workbook
│   └── Sanskrit_Epic_Research_Suite.ipynb <- Unified analysis suite
├── src/
│   └── epic_utils.py             <- Core utility functions for analysis
├── scripts/
│   ├── train_epic_model.py       <- Standalone training script
│   └── evaluate_epic_model.py    <- Model validation and report generation
├── app.py                        <- Gradio deployment interface
└── requirements.txt              <- Dependency specifications
```

## Data Source

The primary data for this project is sourced from:
Muneo Tokunaga's Digital Sanskrit Texts — https://bombay.indology.info/
Electronic texts of the Mahabharata and Ramayana.
