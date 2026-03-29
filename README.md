# FASEROH Symbolic Learning

This project explores a simple but meaningful question: can a neural sequence model learn to translate a symbolic mathematical expression into its Taylor expansion?

For this task, I treat symbolic mathematics as a sequence-to-sequence problem. The input is a function such as `sin(x)` or `exp(x) + x**2`, and the target is its Taylor expansion around `x = 0` up to fourth order. The project combines exact symbolic generation with SymPy and neural modeling with PyTorch, which makes it a good fit for neural-symbolic learning and for the FASEROH GSoC test task.

## What This Project Does

At a high level, the repository does four things:

- generates a synthetic dataset of symbolic functions and their Taylor expansions
- tokenizes and encodes those expressions into sequences
- trains both an LSTM model and a Transformer model
- evaluates predictions using both surface-level and mathematical correctness

The final system is more than a minimal prototype. It includes a full training pipeline, reproducible preprocessing, validation-aware training, and symbolic evaluation.

## Why This Problem Is Interesting

A lot of machine learning work focuses on numbers, labels, or natural language. Symbolic mathematics is different. Here, the output is structured, interpretable, and mathematically constrained. That makes the problem harder, but also much more interesting.

Taylor expansion is a good benchmark for this setting because:

- it is mathematically well defined
- the target expressions are symbolic rather than numeric
- it lets us compare exact syntax and true symbolic equivalence
- it naturally fits a translation-style learning setup

## Core Idea

The system learns mappings of the form:

```text
f(x) -> Taylor(f(x))
```

Some examples:

```text
sin(x)          -> x - x**3/6
exp(x)          -> 1 + x + x**2/2 + x**3/6 + x**4/24
log(1 + x)      -> x - x**2/2 + x**3/3 - x**4/4
sin(x) + exp(x) -> symbolic expansion of the combined function
```

The important part is that the model is not solving a numeric regression problem. It is learning to produce a valid symbolic expression token by token.

## Project Flow

```text
symbolic function
    -> SymPy Taylor expansion
    -> tokenization
    -> vocabulary encoding
    -> padded dataset
    -> seq2seq training
    -> autoregressive decoding
    -> evaluation with token and symbolic metrics
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

## Main Components

### Dataset Generation

File: [src/data/dataset_generator.py](d:/Gsoc/fasero-symbolic-learning/src/data/dataset_generator.py)

This script builds the raw dataset with SymPy. It samples from a small set of symbolic base functions such as:

- `sin(x)`
- `cos(x)`
- `exp(x)`
- `log(1+x)`
- `x`
- `x**2`
- `x**3`

It then combines them through addition and subtraction, computes the Taylor expansion around `x = 0`, and saves the function/expansion pairs to `data/raw/taylor_dataset.json`.

### Tokenization and Vocabulary

Files:
- [src/tokenization/tokenizer.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/tokenizer.py)
- [src/tokenization/vocabulary.py](d:/Gsoc/fasero-symbolic-learning/src/tokenization/vocabulary.py)

Expressions are broken into symbolic tokens such as function names, operators, constants, variables, and punctuation. A shared vocabulary is built over both source and target expressions, with special tokens for:

- `PAD`
- `START`
- `END`

### Preprocessing

File: [src/data/preprocess.py](d:/Gsoc/fasero-symbolic-learning/src/data/preprocess.py)

The preprocessing step:

- tokenizes raw expressions
- encodes them with the vocabulary
- adds start and end markers
- pads them to a fixed maximum length
- stores processed data and metadata in `data/processed/dataset.json`

## Models

### LSTM Seq2Seq

File: [src/models/lstm_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/lstm_seq2seq.py)

The LSTM model is the recurrent baseline. It uses:

- learned embeddings
- stacked LSTM encoder and decoder layers
- scheduled teacher forcing
- dropout
- gradient clipping during training

This model is useful as a solid baseline and as a comparison point for the Transformer.

### Transformer Seq2Seq

File: [src/models/transformer_seq2seq.py](d:/Gsoc/fasero-symbolic-learning/src/models/transformer_seq2seq.py)

The Transformer is the main model in the repository. It uses:

- learned token embeddings
- sinusoidal positional encodings
- multi-head attention
- causal masks for decoding
- padding-aware attention masks

This is the model I would present as the strongest version of the project.

## Training

### LSTM Training

File: [src/training/train_lstm.py](d:/Gsoc/fasero-symbolic-learning/src/training/train_lstm.py)

Features included in the final version:

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

Features included in the final version:

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

The evaluation script supports both models and scores them using:

- token accuracy
- exact match
- symbolic match with SymPy simplification

That last metric is especially important here. Two expressions can look different at the token level but still be mathematically equivalent, so symbolic verification gives a much fairer picture of model quality.

Run:

```bash
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
python src/evaluation/evaluate_model.py --model lstm --num-examples 100
```

## How It Works End To End

1. Raw symbolic expressions are generated with SymPy.
2. Their fourth-order Taylor expansions are computed exactly.
3. Both source and target expressions are tokenized.
4. Tokens are encoded into padded integer sequences.
5. The LSTM or Transformer is trained to predict the target sequence from the input sequence.
6. During inference, the decoder generates one token at a time until it emits `END`.
7. Predictions are compared to targets using token-level and symbolic metrics.

## Final Improvements Added

Compared with the original prototype, the final version is much more complete and much easier to present as serious project work. Improvements include:

- more reliable path handling
- reproducible dataset generation
- validation-aware training
- early stopping
- checkpoint saving
- learning-rate scheduling
- gradient clipping
- label smoothing
- padding-aware embeddings and masks
- cleaner evaluation metrics
- a stronger README and results report

## Recommended Workflow

If you want to run the strongest version of the project from scratch:

```bash
python src/data/dataset_generator.py --num-samples 5000 --series-order 5 --seed 42
python src/data/preprocess.py --max-len 30
python src/training/train_transformer.py
python src/evaluation/evaluate_model.py --model transformer --num-examples 100
```

If you want to compare both architectures:

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

- `data/processed/dataset.json` is committed so reviewers can inspect the processed data format directly.
- Generated checkpoint files such as `lstm_model.pth` and `transformer_model.pth` are ignored via `.gitignore`.
- Final metrics and example outputs are documented in [experiments/results.md](d:/Gsoc/fasero-symbolic-learning/experiments/results.md#L1).

## Tech Stack

- Python
- PyTorch
- SymPy
- NumPy
- tqdm
- scikit-learn
- matplotlib

## Why This Is A Strong GSoC Submission

This project completes the required task, but it also goes further. It does not stop at “train two models.” It includes:

- reproducible dataset generation
- a clean preprocessing pipeline
- two working seq2seq baselines
- meaningful evaluation metrics
- symbolic verification
- stronger training stability
- polished documentation and reporting

In other words, it shows both implementation ability and project maturity.

## Future Directions

If this project were extended further, the most natural next steps would be:

- beam search decoding
- larger and more diverse symbolic datasets
- curriculum learning by expression complexity
- canonical symbolic normalization before evaluation
- additional tasks such as differentiation, simplification, or integration
- experiment dashboards and visualizations

## Closing Note

FASEROH Symbolic Learning is a compact neural-symbolic project with a clear problem, a full working pipeline, and strong final results. The Transformer is the best final model, while the LSTM provides a useful comparison baseline. Together, they make the project both technically complete and easy to explain in a GSoC setting.
