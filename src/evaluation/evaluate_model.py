"""Evaluate trained symbolic seq2seq models and report readable metrics."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import sympy as sp
import torch

from src.evaluation.metrics import exact_match, token_accuracy
from src.models.lstm_seq2seq import Decoder, Encoder, Seq2Seq
from src.models.transformer_seq2seq import TransformerSeq2Seq
from src.tokenization.vocabulary import Vocabulary


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LSTM_MODEL_PATH = PROJECT_ROOT / "lstm_model.pth"
TRANSFORMER_MODEL_PATH = PROJECT_ROOT / "transformer_model.pth"

LSTM_EMBED_SIZE = 128
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
MAX_GENERATION_LENGTH = 30


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate symbolic seq2seq models.")
    parser.add_argument(
        "--model",
        choices=["lstm", "transformer"],
        default="transformer",
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of dataset examples to evaluate.",
    )
    return parser.parse_args()


def load_data(dataset_path: Path):
    with dataset_path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    inputs = torch.tensor(data["inputs"], dtype=torch.long)
    targets = torch.tensor(data["targets"], dtype=torch.long)

    vocab = Vocabulary()
    vocab.token_to_id = data["vocab"]
    vocab.id_to_token = {token_id: token for token, token_id in data["vocab"].items()}

    return inputs, targets, vocab


def build_model(model_name, vocab_size, pad_idx, max_len):
    if model_name == "transformer":
        return TransformerSeq2Seq(vocab_size, pad_idx=pad_idx, max_len=max_len).to(DEVICE)

    encoder = Encoder(
        vocab_size,
        LSTM_EMBED_SIZE,
        LSTM_HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    decoder = Decoder(
        vocab_size,
        LSTM_EMBED_SIZE,
        LSTM_HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    )
    return Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)


def get_model_path(model_name):
    return TRANSFORMER_MODEL_PATH if model_name == "transformer" else LSTM_MODEL_PATH


def predict(model_name, model, src, start_token_id, end_token_id):
    model.eval()
    src = src.unsqueeze(0).to(DEVICE)
    outputs = []

    if model_name == "transformer":
        generated = torch.tensor([[start_token_id]], dtype=torch.long, device=DEVICE)
        for _ in range(MAX_GENERATION_LENGTH):
            with torch.no_grad():
                output = model(src, generated)

            next_token = output[:, -1, :].argmax(dim=1)
            token_id = next_token.item()
            if token_id == end_token_id:
                break

            outputs.append(token_id)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    else:
        with torch.no_grad():
            hidden, cell = model.encoder(src)

        input_token = torch.tensor([start_token_id], dtype=torch.long, device=DEVICE)
        for _ in range(MAX_GENERATION_LENGTH):
            with torch.no_grad():
                output, hidden, cell = model.decoder(input_token, hidden, cell)

            next_token = output.argmax(dim=1)
            token_id = next_token.item()
            if token_id == end_token_id:
                break

            outputs.append(token_id)
            input_token = next_token

    return outputs


def decode(tokens, vocab):
    special_tokens = {"PAD", "START", "END"}
    decoded_tokens = [
        vocab.id_to_token[token_id]
        for token_id in tokens
        if vocab.id_to_token.get(int(token_id)) not in special_tokens
    ]
    return " ".join(decoded_tokens)


def symbolic_match(pred_str, target_str):
    try:
        return sp.simplify(pred_str) == sp.simplify(target_str)
    except Exception:
        return False


def evaluate():
    args = parse_args()
    inputs, targets, vocab = load_data(DATASET_PATH)

    pad_idx = vocab.token_to_id["PAD"]
    start_token_id = vocab.token_to_id["START"]
    end_token_id = vocab.token_to_id["END"]
    max_len = inputs.shape[1]

    model_path = get_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = build_model(args.model, len(vocab.token_to_id), pad_idx, max_len)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    num_examples = min(args.num_examples, len(inputs))
    total_acc = 0.0
    total_em = 0.0
    total_sym = 0.0

    for index in range(num_examples):
        src = inputs[index]
        target_tokens = targets[index].tolist()
        pred_tokens = predict(args.model, model, src, start_token_id, end_token_id)

        pred_str = decode(pred_tokens, vocab)
        target_str = decode(target_tokens, vocab)

        acc = token_accuracy(pred_tokens, target_tokens, special_token_ids={pad_idx, start_token_id, end_token_id})
        em = exact_match(pred_tokens, target_tokens, special_token_ids={pad_idx, start_token_id, end_token_id})
        sym = symbolic_match(pred_str, target_str)

        total_acc += acc
        total_em += em
        total_sym += float(sym)

        if index < 10:
            print(f"\nExample {index + 1}")
            print("Predicted     :", pred_str)
            print("Target        :", target_str)
            print(f"Token Accuracy: {acc:.4f}")
            print("Exact Match   :", em)
            print("Symbolic Match:", sym)

    print("\n==== SUMMARY ====")
    print("Model          :", args.model)
    print("Examples       :", num_examples)
    print(f"Avg Token Acc  : {total_acc / num_examples:.4f}")
    print(f"Exact Match    : {total_em / num_examples:.4f}")
    print(f"Symbolic Match : {total_sym / num_examples:.4f}")


if __name__ == "__main__":
    evaluate()
