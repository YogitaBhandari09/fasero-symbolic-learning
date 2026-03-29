# FASEROH Symbolic Learning

What happens if we treat symbolic mathematics like translation?

That is the core idea behind this repository: take a symbolic expression as input, and generate its Taylor expansion as an output sequence. It is a small problem compared to the full ML4SCI FASEROH setting, but it captures the part I found most interesting while building the project: once the model has to produce a symbolic formula, token-level accuracy stops being enough and mathematical correctness starts to matter.

## Quick Summary

- **Task:** learn fourth-order Taylor expansions of symbolic expressions around `x = 0`
- **Models:** LSTM seq2seq baseline and Transformer seq2seq model
- **Main takeaway:** both models learn the controlled task well, but the Transformer is the more convincing choice once the symbolic space becomes richer
- **Evaluation:** token accuracy, exact match, and symbolic equivalence with SymPy
- **Why this matters for FASEROH:** the current project focuses on the symbolic decoding side of the larger histogram-to-symbolic mapping problem

## Quick Start

If you want to run the full pipeline end to end:

```bash
pip install -r requirements.txt
python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

To compare both models:

```bash
python src/training/train_lstm.py
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

## Why I Built This

I wanted a project that was small enough to finish properly, but still close enough to the kind of reasoning FASEROH cares about. Taylor expansion turned out to be a good middle ground. It is symbolic, structured, easy to validate exactly, and still hard enough that model behavior is visible when something goes wrong.

One practical lesson from building it was that symbolic tasks expose shortcuts quickly. A model can look good on surface metrics while still failing in mathematically obvious ways. That pushed me to make symbolic evaluation a first-class part of the project instead of treating it like a nice extra.

## Problem Setup

The learning task is:

```text
f(x) -> Taylor(f(x))
```

Examples:

```text
sin(x)          -> x - x**3/6
exp(x)          -> 1 + x + x**2/2 + x**3/6 + x**4/24
log(1 + x)      -> x - x**2/2 + x**3/3 - x**4/4
sin(x) + exp(x) -> symbolic expansion of the combined function
```

I frame this as sequence-to-sequence learning:

- the source sequence is the tokenized symbolic input
- the target sequence is the tokenized Taylor expansion
- the decoder generates the target one token at a time

## Why This Matters

Most machine learning projects only need the model to predict a number, a class, or a continuous vector. Symbolic learning is different. The output has syntax, structure, and mathematical meaning. That makes it closer to how equations are actually used in physics and scientific modeling.

In the broader FASEROH context, the real interest is not Taylor expansion by itself. The more interesting goal is recovering symbolic structure from scientific data. That is why I treated this repository as more than a toy benchmark: it is a clean way to study the symbolic generation side of that problem.

## Key Contributions

- Built a complete symbolic data generation pipeline with SymPy
- Implemented tokenization, vocabulary construction, and sequence preprocessing
- Trained and compared two sequence models: LSTM and Transformer
- Added symbolic evaluation using SymPy simplification
- Made the pipeline reproducible and easy to rerun from scratch
- Documented the project in a way that makes the design choices inspectable

## System Overview

```text
symbolic expression
    -> exact Taylor expansion with SymPy
    -> tokenization
    -> vocabulary encoding
    -> padded dataset
    -> seq2seq training
    -> autoregressive decoding
    -> symbolic evaluation
```

## Repository Structure

```text
fasero-symbolic-learning/
|-- data/
|   |-- raw/
|   `-- processed/
|-- experiments/
|-- notebooks/
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- models/
|   |-- tokenization/
|   |-- training/
|   `-- utils/
|-- README.md
|-- requirements.txt
`-- run.sh
```

## Data Pipeline

### Symbolic Dataset Generation

File: [src/data/dataset_generator.py](d:/Gsoc/fasero-symbolic-learning/src/data/dataset_generator.py)

The dataset is generated with SymPy from a small family of base functions:

- `sin(x)`
- `cos(x)`
- `exp(x)`
- `log(1+x)`
- `x`
- `x**2`
- `x**3`

These are combined through addition and subtraction and then expanded around `x = 0`. I kept the function family intentionally small at first because it made debugging much easier. When a prediction failed, it was still possible to inspect the expression by hand and see whether the issue came from tokenization, data generation, or decoding.

### Tokenization and Vocabulary

Files:
- [src/tokenization/tokenizer.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/tokenizer.py)
- [src/tokenization/vocabulary.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/vocabulary.py)

Expressions are split into symbolic tokens such as:

- function names: `sin`, `cos`, `exp`, `log`
- operators: `+`, `-`, `*`, `**`, `/`
- variables and constants: `x`, `1`, `2`, `3`
- punctuation: `(`, `)`

The vocabulary is shared across source and target sequences and includes:

- `PAD`
- `START`
- `END`

### Preprocessing

File: [src/data/preprocess.py](d:/Gsoc/fasero-symbolic-learning/src/data/preprocess.py)

The preprocessing stage:

- tokenizes raw expressions
- converts tokens to integer IDs
- adds start and end markers
- pads sequences to a fixed length
- stores the processed dataset in `data/processed/dataset.json`

## Models

### LSTM Seq2Seq

File: [src/models/lstm_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/lstm_seq2seq.py)

The LSTM model is the recurrent baseline. It uses:

- learned token embeddings
- stacked encoder and decoder LSTMs
- scheduled teacher forcing
- dropout
- gradient clipping during training

I kept this model in the project because a baseline matters. On a controlled symbolic task, it is useful to know whether the Transformer is helping for the right reasons or whether the dataset is simply easy enough that almost any decent seq2seq model can do well.

### Transformer Seq2Seq

File: [src/models/transformer_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/transformer_seq2seq.py)

The Transformer is the main model in the repository. It uses:

- learned token embeddings
- sinusoidal positional encoding
- multi-head attention
- causal decoding masks
- padding-aware attention masks

This is the model I would choose as the primary architecture for any extension of the project. It handles longer dependencies more cleanly and is a better match for more open-ended symbolic generation.

## Training Setup

### LSTM Training

File: [src/training/train_lstm.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_lstm.py)

The LSTM training pipeline includes:

- train/validation split
- AdamW optimizer
- scheduled teacher forcing
- label smoothing
- gradient clipping
- early stopping
- best-checkpoint saving

Run:

```bash
python src/training/train_lstm.py
```

### Transformer Training

File: [src/training/train_transformer.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_transformer.py)

The Transformer training pipeline includes:

- train/validation split
- AdamW optimizer
- learning-rate scheduling
- label smoothing
- gradient clipping
- early stopping
- best-checkpoint saving

Run:

```bash
python src/training/train_transformer.py
```

## Evaluation

File: [src/evaluation/evaluate_model.py](d:/Gsoc/fasero-symbolic-learning/src/evaluation/evaluate_model.py)

The evaluation script reports:

- token accuracy
- exact match
- symbolic match through SymPy simplification

For this kind of task, symbolic match is the most informative metric. Two expressions may differ token by token and still represent the same function. That happened often enough during debugging that I stopped trusting exact-match results by themselves.

Run:

```bash
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
```

## Results

On the current evaluation slice, both models score very highly. I do not interpret that as “the problem is solved.” The dataset is only 5000 examples, the function family is narrow, and many patterns repeat in related forms. Under those conditions, high scores are expected.

The more realistic reading is:

- the models are learning the symbolic mapping successfully on a controlled dataset
- the current benchmark is not difficult enough to separate the two models sharply
- the Transformer still looks like the better long-term choice once the expressions become more diverse and less templated

That last point matters. Even when the headline scores are similar, the Transformer is still the model I trust more for a harder version of the task because:

1. it handles longer-range symbolic dependencies more naturally
2. it should scale better as the function space becomes richer
3. it is a closer fit to the sequence-generation setting that FASEROH will eventually require

Detailed examples and the current evaluation summary are in [experiments/results.md](d:/Gsoc/fasero-symbolic-learning/experiments/results.md#L1).

## Extension Toward Histogram-to-Symbolic Mapping

This repository is not the full FASEROH task, but it is closely related to it.

The larger ML4SCI FASEROH direction is closer to:

```text
histogram or distribution-like input
    -> learned encoder representation
    -> sequence decoder / transformer
    -> symbolic function
```

That can be viewed as:

- **histogram:** a structured sequence of binned information
- **encoder:** learns a useful latent representation of that sequence
- **decoder:** generates a symbolic expression token by token

The current project already covers much of the symbolic side of that pipeline:

- symbolic output representation
- autoregressive decoding
- symbolic evaluation
- model comparison between recurrent and attention-based approaches

In other words, this repository does not solve the histogram-input side yet, but it already tackles the part that is usually harder to evaluate cleanly: generating and validating symbolic outputs.

## Limitations

The current version has some clear limitations:

- the dataset is small
- the function space is narrow
- the decoder has no explicit grammar constraints
- the evaluation slice is easy enough that overfitting is a real possibility
- symbolic validity is checked after decoding rather than enforced during decoding

I think being explicit about these limitations is important. The project works, but the current benchmark is still controlled and modest in scope.

## Proposed Improvements

The next steps I would prioritize are:

- grammar-constrained decoding
- beam search instead of greedy decoding
- a functional loss based on numerical consistency at sampled points
- larger and more diverse symbolic datasets
- curriculum learning by expression complexity
- extension from symbolic-sequence input to histogram-based input

Among these, the histogram-input extension and grammar-aware decoding feel the most important if the goal is to move closer to the actual FASEROH setting.

## Reproducibility

To run the main pipeline from scratch:

```bash
python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

To compare both models:

```bash
python src/training/train_lstm.py
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

## Installation

```bash
pip install -r requirements.txt
```

## Repository Notes

- `data/processed/dataset.json` is included so reviewers can inspect the processed data format directly
- checkpoint files such as `lstm_model.pth` and `transformer_model.pth` are ignored via `.gitignore`
- final metrics and example outputs are summarized in [experiments/results.md](d:/Gsoc/fasero-symbolic-learning/experiments/results.md#L1)

## Tech Stack

- Python
- PyTorch
- SymPy
- NumPy
- tqdm
- scikit-learn
- matplotlib

## Closing Remarks

I built this project to be small enough to finish carefully, but still serious enough to say something useful about symbolic generation. It helped me understand where sequence modeling works well, where evaluation becomes tricky, and why symbolic correctness has to be treated differently from ordinary token prediction.

As a standalone repository, it is a compact neural-symbolic benchmark. As preparation for FASEROH, it helped me focus on the part of the problem where the model has to produce a mathematically meaningful symbolic expression rather than just a numerical answer.
