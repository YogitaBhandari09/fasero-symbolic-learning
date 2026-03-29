"""Tokenize and encode the raw symbolic dataset into padded sequences."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tokenization.tokenizer import tokenize
from src.tokenization.vocabulary import Vocabulary


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "taylor_dataset.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"


def parse_args():
    """Read preprocessing settings from the command line."""
    parser = argparse.ArgumentParser(description="Tokenize and encode symbolic Taylor-series data.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=30,
        help="Maximum padded sequence length including START and END tokens.",
    )
    return parser.parse_args()


def pad_sequence(sequence, max_len, pad_id):
    """Pad a token-id sequence up to the requested length."""
    return sequence + [pad_id] * (max_len - len(sequence))


def main():
    """Build the processed dataset used by the training scripts."""
    args = parse_args()

    with RAW_DATA_PATH.open(encoding="utf-8") as raw_file:
        raw_samples = json.load(raw_file)

    source_token_lists = []
    target_token_lists = []

    for sample in raw_samples:
        source_token_lists.append(tokenize(sample["function"]))
        target_token_lists.append(tokenize(sample["taylor"]))

    vocabulary = Vocabulary()
    vocabulary.build(source_token_lists + target_token_lists)

    encoded_inputs = []
    encoded_targets = []
    observed_max_input_len = 0
    observed_max_target_len = 0

    for source_tokens, target_tokens in zip(source_token_lists, target_token_lists):
        source_ids = [vocabulary.token_to_id["START"]] + vocabulary.encode(source_tokens) + [vocabulary.token_to_id["END"]]
        target_ids = [vocabulary.token_to_id["START"]] + vocabulary.encode(target_tokens) + [vocabulary.token_to_id["END"]]

        observed_max_input_len = max(observed_max_input_len, len(source_ids))
        observed_max_target_len = max(observed_max_target_len, len(target_ids))

        if len(source_ids) > args.max_len or len(target_ids) > args.max_len:
            raise ValueError(
                f"Found sequence longer than max_len={args.max_len}. "
                f"Observed lengths: input={len(source_ids)}, target={len(target_ids)}."
            )

        encoded_inputs.append(pad_sequence(source_ids, args.max_len, vocabulary.token_to_id["PAD"]))
        encoded_targets.append(pad_sequence(target_ids, args.max_len, vocabulary.token_to_id["PAD"]))

    processed_dataset = {
        "inputs": encoded_inputs,
        "targets": encoded_targets,
        "vocab": vocabulary.token_to_id,
        "metadata": {
            "max_len": args.max_len,
            "num_samples": len(encoded_inputs),
            "observed_max_input_len": observed_max_input_len,
            "observed_max_target_len": observed_max_target_len,
        },
    }

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_DATA_PATH.open("w", encoding="utf-8") as processed_file:
        json.dump(processed_dataset, processed_file)

    print(
        "Preprocessing complete. "
        f"Samples={len(encoded_inputs)}, vocab_size={len(vocabulary.token_to_id)}, "
        f"max_len={args.max_len}"
    )


if __name__ == "__main__":
    main()
