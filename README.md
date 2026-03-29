# FASEROH Symbolic Learning

Neural sequence-to-sequence learning for symbolic mathematics, focused on mapping input expressions to their Taylor-series expansions. This project combines symbolic data generation with deep learning models so that algebraic structure can be learned as a translation problem.

## Overview

This repository treats symbolic reasoning as a seq2seq task:

- Input: a symbolic mathematical expression such as `sin(x) + x**2`
- Output: its Taylor expansion around `x = 0`
- Objective: learn token-level symbolic translation with strong exact-sequence and symbolic-equivalence performance

The project includes:

- Synthetic dataset generation using SymPy
- Tokenization and vocabulary building for symbolic expressions
- An upgraded LSTM encoder-decoder baseline
- An upgraded Transformer encoder-decoder model
- Evaluation using token accuracy, exact match, and symbolic equivalence
- A complete training and inference pipeline

## Why This Project Matters

Symbolic learning sits at the intersection of:

- Deep learning
- program induction
- mathematical reasoning
- scientific AI
- neural-symbolic systems

Instead of predicting only numeric values, this project predicts structured symbolic expressions. That makes it closer to mathematical reasoning, computer algebra, and interpretable AI than a standard regression or classification pipeline.

## Core Idea

We generate paired examples of the form:

```text
f(x) -> Taylor(f(x))
```

Examples:

```text
sin(x)          -> x - x**3/6
exp(x)          -> 1 + x + x**2/2 + x**3/6 + x**4/24
log(1 + x)      -> x - x**2/2 + x**3/3 - x**4/4
sin(x) + exp(x) -> combined symbolic expansion
```

The model learns to translate symbolic syntax into another symbolic syntax, similar to machine translation, but over mathematical expressions rather than natural language.

## Project Architecture

```text
raw symbolic expressions
    -> tokenization
    -> vocabulary encoding
    -> padded sequence dataset
    -> seq2seq training
    -> auto-regressive decoding
    -> symbolic evaluation with SymPy
```

### Repository Structure

```text
fasero-symbolic-learning/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
├── notebooks/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── tokenization/
│   ├── training/
│   └── utils/
├── README.md
├── requirements.txt
└── run.sh
```

## Models

### 1. LSTM Seq2Seq

The LSTM model is a recurrent encoder-decoder architecture for symbolic translation.

Key characteristics:

- learned token embeddings with padding awareness
- stacked LSTM encoder and decoder
- scheduled teacher forcing during training
- dropout regularization
- gradient clipping for stability

This model is a strong baseline and is useful for showing how much improvement the Transformer gives.

### 2. Transformer Seq2Seq

The Transformer is the flagship model in the repository.

Key characteristics:

- multi-head self-attention
- sinusoidal positional encoding
- causal target masking for autoregressive decoding
- source and target padding masks
- AdamW optimization
- validation-aware training with checkpointing and early stopping

This is the best model in the project for most serious experiments and should be treated as the default architecture for final results.

## Improvements Added In The Final Version

The project now includes a much stronger training and evaluation stack than the original prototype:

- robust path handling so scripts work from the repo root or direct file execution
- validation split for both training pipelines
- early stopping based on validation loss
- best-model checkpoint saving
- learning-rate scheduling with `ReduceLROnPlateau`
- gradient clipping
- label smoothing
- padding-aware embeddings
- improved Transformer masking
- cleaner sequence metrics that ignore special tokens
- model-selectable evaluation CLI
- reproducible dataset generation with seed control
- safer preprocessing with metadata and sequence-length checks

## Tech Stack

### Languages and Libraries

- Python
- PyTorch
- SymPy
- NumPy
- tqdm
- scikit-learn
- matplotlib

### What Each Technology Does

- `PyTorch`: defines, trains, and evaluates the neural seq2seq models
- `SymPy`: generates symbolic expressions and verifies symbolic equivalence
- `NumPy`: general numerical support
- `tqdm`: progress bars during dataset generation and long-running jobs
- `scikit-learn`: available for future metrics and dataset analysis workflows
- `matplotlib`: available for visualization and experiment reporting

## Data Pipeline

### Step 1. Dataset Generation

File: [src/data/dataset_generator.py](d:/Gsoc/fasero-symbolic-learning/src/data/dataset_generator.py)

This script:

- samples symbolic base functions such as `sin(x)`, `cos(x)`, `exp(x)`, `log(1+x)`, `x`, `x**2`, `x**3`
- composes them through addition and subtraction
- computes Taylor expansions using SymPy
- stores paired samples in `data/raw/taylor_dataset.json`

### Step 2. Tokenization

File: [src/tokenization/tokenizer.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/tokenizer.py)

Expressions are split into symbolic tokens such as:

- operators: `+`, `-`, `*`, `**`, `/`
- functions: `sin`, `cos`, `exp`, `log`
- variables and constants: `x`, `1`, `2`, `3`, `24`
- punctuation: `(`, `)`

### Step 3. Vocabulary Building

File: [src/tokenization/vocabulary.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/vocabulary.py)

The vocabulary maps symbolic tokens to IDs and reserves:

- `PAD`
- `START`
- `END`

### Step 4. Preprocessing

File: [src/data/preprocess.py](d:/Gsoc/fasero-symbolic-learning/src/data/preprocess.py)

This script:

- tokenizes raw expressions
- builds a unified vocabulary
- adds start/end tokens
- pads sequences to a fixed maximum length
- stores processed tensors plus metadata in `data/processed/dataset.json`

## Training Pipeline

### LSTM Training

File: [src/training/train_lstm.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_lstm.py)

Training features:

- train/validation split
- scheduled teacher forcing
- AdamW optimizer
- label smoothing
- gradient clipping
- early stopping
- best checkpoint saving to `lstm_model.pth`

Run:

```bash
python src/training/train_lstm.py
```

### Transformer Training

File: [src/training/train_transformer.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_transformer.py)

Training features:

- train/validation split
- AdamW optimizer
- padding-aware attention masks
- label smoothing
- gradient clipping
- early stopping
- best checkpoint saving to `transformer_model.pth`

Run:

```bash
python src/training/train_transformer.py
```

## Evaluation Pipeline

File: [src/evaluation/evaluate_model.py](d:/Gsoc/fasero-symbolic-learning/src/evaluation/evaluate_model.py)

The evaluation script supports both architectures and performs autoregressive decoding before scoring predictions.

Metrics:

- token accuracy
- exact sequence match
- symbolic equivalence using SymPy simplification

Run:

```bash
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
```

## How The System Works End To End

1. `dataset_generator.py` creates raw symbolic function/expansion pairs.
2. `preprocess.py` tokenizes them, builds a vocabulary, and encodes sequences.
3. `train_lstm.py` or `train_transformer.py` trains a seq2seq model on the processed dataset.
4. During inference, the decoder starts from `START` and predicts one token at a time until `END`.
5. `evaluate_model.py` converts predicted token IDs back to symbolic strings.
6. Metrics compare predicted output to the target both syntactically and symbolically.

## Recommended Best Workflow

For the strongest project presentation and best expected results:

```bash
python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

If you want to compare baselines:

```bash
python src/training/train_lstm.py
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

Current `requirements.txt`:

- torch
- numpy
- sympy
- tqdm
- scikit-learn
- matplotlib

## Research and GSoC Value

This project demonstrates:

- neural translation over symbolic programs
- a reproducible data-generation to evaluation pipeline
- comparison between recurrent and attention-based architectures
- symbolic verification beyond surface token matching
- an interpretable neural-symbolic workflow suitable for academic extension

It is a strong GSoC-style project because it combines:

- mathematical depth
- machine learning engineering
- symbolic AI
- reproducibility
- research-facing evaluation

## Future Extensions

The most promising next upgrades would be:

- beam search decoding
- larger and more diverse symbolic datasets
- curriculum learning by expression complexity
- exact symbolic canonicalization before scoring
- attention visualizations
- experiment tracking and result dashboards
- support for integration, differentiation, simplification, or equation solving tasks

## Final Summary

FASEROH Symbolic Learning is a neural-symbolic seq2seq system for learning symbolic Taylor expansions from mathematical expressions. It uses SymPy for data generation and symbolic verification, PyTorch for deep learning, and both LSTM and Transformer architectures for sequence modeling. The final upgraded version of the project includes stronger training stability, better evaluation, reproducible preprocessing, and a cleaner research-grade project structure.

If you want the best model in this repository, use the Transformer pipeline as the primary result model and keep the LSTM as the comparative baseline.
