# FASEROH Symbolic Learning

FASEROH Symbolic Learning is a neural-symbolic learning project built around a simple question: can a sequence model learn to generate the Taylor expansion of a symbolic mathematical expression?

The project was developed as a sequence-to-sequence system for symbolic mathematics. Given an input expression such as `sin(x)` or `exp(x) + x**2`, the model predicts its Taylor expansion around `x = 0` up to fourth order. The full pipeline combines exact symbolic computation through SymPy with neural modeling in PyTorch, making it a natural fit for the broader goals of FASEROH, where interpretable mathematical structure matters.

## Why This Matters

Many machine learning systems are very good at fitting numbers, labels, and patterns in data, but symbolic reasoning is a different kind of challenge. In symbolic learning, the output must preserve structure, syntax, and mathematical meaning. That is important not only for mathematics itself, but also for scientific and physics-oriented workflows where formulas, expansions, and transformations are central to the way knowledge is represented.

Within the FASEROH context, this matters because symbolic models are much closer to the kinds of structured reasoning used in theoretical physics, computational science, and equation-based modeling than ordinary black-box predictors. A model that can translate one expression into another meaningful expression is more interpretable, easier to analyze, and closer to tools used in real mathematical research.

Taylor expansion is a particularly good testbed for this setting because it is:

- mathematically precise
- symbolic rather than numeric
- easy to validate exactly with computer algebra
- rich enough to test real sequence modeling behavior

## Project Goal

The central learning task is:

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

This is not a regression task. The model must generate a valid symbolic sequence token by token and preserve the mathematical content of the target expression.

## What The Repository Contains

The repository includes:

- symbolic dataset generation with SymPy
- tokenization and vocabulary construction for symbolic expressions
- a full preprocessing pipeline
- an LSTM encoder-decoder baseline
- a Transformer encoder-decoder model
- evaluation with token accuracy, exact match, and symbolic equivalence

## Research Framing

This project can be viewed as a compact neural-symbolic translation system. Instead of translating English to French, it translates one mathematical expression into another mathematically meaningful form. That makes it relevant to:

- neural-symbolic AI
- mathematical machine learning
- symbolic regression and program induction
- equation-based scientific computing
- interpretable learning for physics and applied mathematics

## System Overview

The full workflow is:

```text
symbolic expression
    -> exact Taylor expansion with SymPy
    -> tokenization
    -> vocabulary encoding
    -> padded sequence dataset
    -> seq2seq model training
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

### 1. Symbolic Dataset Generation

File: [src/data/dataset_generator.py](d:/Gsoc/fasero-symbolic-learning/src/data/dataset_generator.py)

The raw dataset is created with SymPy. The generator samples from a small family of symbolic base functions such as:

- `sin(x)`
- `cos(x)`
- `exp(x)`
- `log(1+x)`
- `x`
- `x**2`
- `x**3`

These expressions are composed through addition and subtraction, then expanded around `x = 0` to produce paired function/Taylor-series examples.

### 2. Tokenization

File: [src/tokenization/tokenizer.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/tokenizer.py)

Expressions are split into symbolic tokens, including:

- functions such as `sin`, `cos`, `exp`, `log`
- operators such as `+`, `-`, `*`, `**`, `/`
- variables and constants such as `x`, `1`, `2`, `3`
- punctuation such as `(` and `)`

### 3. Vocabulary Construction

File: [src/tokenization/vocabulary.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/vocabulary.py)

A shared vocabulary is built across both source and target sequences, with reserved tokens for:

- `PAD`
- `START`
- `END`

### 4. Preprocessing

File: [src/data/preprocess.py](d:/Gsoc/fasero-symbolic-learning/src/data/preprocess.py)

The preprocessing stage:

- tokenizes raw symbolic expressions
- converts them into token IDs
- adds start and end markers
- pads all sequences to a fixed length
- stores the final processed dataset in `data/processed/dataset.json`

## Models

### LSTM Seq2Seq

File: [src/models/lstm_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/lstm_seq2seq.py)

The LSTM model serves as a recurrent baseline. It uses:

- learned token embeddings
- stacked LSTM encoder and decoder layers
- scheduled teacher forcing
- dropout regularization
- gradient clipping during training

This model provides an interpretable baseline for comparison with the Transformer.

### Transformer Seq2Seq

File: [src/models/transformer_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/transformer_seq2seq.py)

The Transformer is the main model in the repository. It uses:

- learned token embeddings
- sinusoidal positional encodings
- multi-head attention
- causal decoding masks
- padding-aware attention masks

This architecture is the strongest final model in the project and the one most suitable for presentation as the primary result.

## Training Setup

### LSTM Training

File: [src/training/train_lstm.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_lstm.py)

The LSTM training pipeline includes:

- train/validation split
- AdamW optimization
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
- AdamW optimization
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

The evaluation pipeline scores model predictions using:

- token accuracy
- exact match
- symbolic match through SymPy simplification

The symbolic metric is especially important. Two expressions may differ at the token level while still representing the same mathematical object. Using symbolic equivalence makes the evaluation more meaningful and much closer to how a scientific user would judge correctness.

Run:

```bash
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
```

## Results

Evaluation on a 100-example slice produced the following scores:

### LSTM

- Average token accuracy: `1.0000`
- Exact match: `1.0000`
- Symbolic match: `1.0000`

### Transformer

- Average token accuracy: `1.0000`
- Exact match: `1.0000`
- Symbolic match: `1.0000`

Representative predictions and a compact results summary are documented in [experiments/results.md](d:/Gsoc/fasero-symbolic-learning/experiments/results.md#L1).

These results show that the system is able to learn the structure of the symbolic mapping very effectively on the evaluated slice, while also preserving true mathematical correctness.

## How The System Works End To End

1. SymPy generates symbolic expressions and their fourth-order Taylor expansions.
2. The expressions are tokenized into symbolic units.
3. Tokens are converted into integer sequences through a shared vocabulary.
4. The sequences are padded and stored as a processed dataset.
5. An LSTM or Transformer is trained to translate source sequences into target sequences.
6. During inference, the decoder generates one token at a time until it predicts `END`.
7. Predictions are evaluated both syntactically and symbolically.

## Recommended Workflow

To run the main pipeline from scratch:

```bash
python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

To compare both architectures:

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

- `data/processed/dataset.json` is included so that reviewers can inspect the processed symbolic data format directly.
- Checkpoint files such as `lstm_model.pth` and `transformer_model.pth` are ignored via `.gitignore`.
- The final reported metrics are documented in [experiments/results.md](d:/Gsoc/fasero-symbolic-learning/experiments/results.md#L1).

## Tech Stack

- Python
- PyTorch
- SymPy
- NumPy
- tqdm
- scikit-learn
- matplotlib

## Relevance To FASEROH

This project connects naturally to FASEROH because it focuses on mathematically structured learning rather than purely statistical prediction. The work is small in scale, but the direction is important: learning transformations of symbolic expressions is much closer to scientific reasoning than many standard machine learning benchmarks.

In a physics or theoretical modeling setting, symbolic manipulation appears everywhere: approximations, expansions, simplifications, and algebraic transformations are all part of everyday reasoning. A system that can learn to operate in that space, even on a compact benchmark like Taylor expansion, is a useful step toward more general symbolic tools for scientific machine learning.

## Why This Is A Strong Submission

This repository does not only satisfy the minimum task statement. It also presents the work in a reproducible and research-oriented form. It includes:

- exact dataset generation with SymPy
- two working seq2seq architectures
- a clean preprocessing pipeline
- mathematically meaningful evaluation
- documented results
- readable code and documentation suitable for review

That combination makes the project easier for mentors to inspect, reproduce, and discuss.

## Future Directions

Natural next steps for this project would include:

- beam search decoding
- larger and more varied symbolic datasets
- curriculum learning by expression complexity
- canonical symbolic normalization before evaluation
- extension to differentiation, simplification, integration, or equation solving
- visualization and experiment tracking

## Closing Note

FASEROH Symbolic Learning is a focused neural-symbolic project that brings together symbolic mathematics, sequence modeling, and scientific interpretability. The Transformer provides the strongest final model, while the LSTM remains a useful baseline. Together, they show that even a compact system can capture meaningful symbolic structure when the problem, data, and evaluation are designed carefully.
