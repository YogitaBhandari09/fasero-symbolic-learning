import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.lstm_seq2seq import Decoder, Encoder, Seq2Seq


DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.json"
MODEL_PATH = PROJECT_ROOT / "lstm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
EMBED_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 20
VALIDATION_SPLIT = 0.1
GRAD_CLIP = 1.0
PATIENCE = 5
MIN_TEACHER_FORCING = 0.2


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path: Path) -> tuple[TensorDataset, dict[str, int]]:
    with dataset_path.open(encoding="utf-8") as dataset_file:
        data = json.load(dataset_file)

    inputs = torch.tensor(data["inputs"], dtype=torch.long)
    targets = torch.tensor(data["targets"], dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    return dataset, data["vocab"]


def build_model(vocab_size: int, pad_idx: int) -> Seq2Seq:
    encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx=pad_idx, num_layers=NUM_LAYERS, dropout=DROPOUT)
    decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, pad_idx=pad_idx, num_layers=NUM_LAYERS, dropout=DROPOUT)
    return Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)


def create_data_loaders(dataset: TensorDataset):
    val_size = max(1, int(len(dataset) * VALIDATION_SPLIT))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, vocab_size, teacher_forcing_ratio, train_mode):
    model.train(mode=train_mode)
    total_loss = 0.0

    for src, trg in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
            logits = output[:, 1:].reshape(-1, vocab_size)
            target_tokens = trg[:, 1:].reshape(-1)
            loss = criterion(logits, target_tokens)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train() -> None:
    set_seed(SEED)
    dataset, vocab = load_dataset(DATASET_PATH)
    pad_idx = vocab["PAD"]
    vocab_size = len(vocab)

    train_loader, val_loader = create_data_loaders(dataset)
    model = build_model(vocab_size, pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        teacher_forcing_ratio = max(MIN_TEACHER_FORCING, 1.0 - (epoch / NUM_EPOCHS) * 0.6)
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, vocab_size, teacher_forcing_ratio, train_mode=True
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model, val_loader, criterion, optimizer, vocab_size, 0.0, train_mode=False
            )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"teacher_forcing={teacher_forcing_ratio:.2f} | lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best LSTM model to {MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
