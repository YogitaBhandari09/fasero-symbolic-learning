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
    parser = argparse.ArgumentParser(description="Tokenize and encode symbolic Taylor-series data.")
    parser.add_argument(
        "--max-len",
        type=int,
        default=30,
        help="Maximum padded sequence length including START and END tokens.",
    )
    return parser.parse_args()


def pad_sequence(sequence, max_len, pad_id):
    return sequence + [pad_id] * (max_len - len(sequence))


def main():
    args = parse_args()

    with RAW_DATA_PATH.open(encoding="utf-8") as raw_file:
        data = json.load(raw_file)

    input_tokens = []
    target_tokens = []

    for sample in data:
        input_tokens.append(tokenize(sample["function"]))
        target_tokens.append(tokenize(sample["taylor"]))

    vocab = Vocabulary()
    vocab.build(input_tokens + target_tokens)

    encoded_inputs = []
    encoded_targets = []
    observed_max_input_len = 0
    observed_max_target_len = 0

    for inp, tgt in zip(input_tokens, target_tokens):
        inp_ids = [vocab.token_to_id["START"]] + vocab.encode(inp) + [vocab.token_to_id["END"]]
        tgt_ids = [vocab.token_to_id["START"]] + vocab.encode(tgt) + [vocab.token_to_id["END"]]

        observed_max_input_len = max(observed_max_input_len, len(inp_ids))
        observed_max_target_len = max(observed_max_target_len, len(tgt_ids))

        if len(inp_ids) > args.max_len or len(tgt_ids) > args.max_len:
            raise ValueError(
                f"Found sequence longer than max_len={args.max_len}. "
                f"Observed lengths: input={len(inp_ids)}, target={len(tgt_ids)}."
            )

        encoded_inputs.append(pad_sequence(inp_ids, args.max_len, vocab.token_to_id["PAD"]))
        encoded_targets.append(pad_sequence(tgt_ids, args.max_len, vocab.token_to_id["PAD"]))

    processed = {
        "inputs": encoded_inputs,
        "targets": encoded_targets,
        "vocab": vocab.token_to_id,
        "metadata": {
            "max_len": args.max_len,
            "num_samples": len(encoded_inputs),
            "observed_max_input_len": observed_max_input_len,
            "observed_max_target_len": observed_max_target_len,
        },
    }

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_DATA_PATH.open("w", encoding="utf-8") as processed_file:
        json.dump(processed, processed_file)

    print(
        "Preprocessing complete. "
        f"Samples={len(encoded_inputs)}, vocab_size={len(vocab.token_to_id)}, "
        f"max_len={args.max_len}"
    )


if __name__ == "__main__":
    main()
