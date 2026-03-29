"""Train the LSTM baseline for symbolic Taylor expansion generation."""

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
    """Set the random seed for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path: Path) -> tuple[TensorDataset, dict[str, int]]:
    """Load the processed symbolic dataset from disk."""
    with dataset_path.open(encoding="utf-8") as dataset_file:
        dataset_payload = json.load(dataset_file)

    input_tensor = torch.tensor(dataset_payload["inputs"], dtype=torch.long)
    target_tensor = torch.tensor(dataset_payload["targets"], dtype=torch.long)
    dataset = TensorDataset(input_tensor, target_tensor)
    return dataset, dataset_payload["vocab"]


def build_model(vocab_size: int, pad_idx: int) -> Seq2Seq:
    """Construct the LSTM encoder-decoder model."""
    encoder = Encoder(
        vocab_size,
        EMBED_SIZE,
        HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    decoder = Decoder(
        vocab_size,
        EMBED_SIZE,
        HIDDEN_SIZE,
        pad_idx=pad_idx,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )
    return Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)


def create_data_loaders(dataset: TensorDataset):
    """Split the dataset into train and validation loaders."""
    validation_size = max(1, int(len(dataset) * VALIDATION_SPLIT))
    training_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(SEED)
    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=generator,
    )

    training_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return training_loader, validation_loader


def run_epoch(model, data_loader, criterion, optimizer, vocab_size, teacher_forcing_ratio, train_mode):
    """Run one full pass over a loader and return the mean loss."""
    model.train(mode=train_mode)
    total_loss = 0.0

    for source_batch, target_batch in data_loader:
        source_batch = source_batch.to(DEVICE)
        target_batch = target_batch.to(DEVICE)

        with torch.set_grad_enabled(train_mode):
            model_output = model(
                source_batch,
                target_batch,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            logits = model_output[:, 1:].reshape(-1, vocab_size)
            expected_tokens = target_batch[:, 1:].reshape(-1)
            loss = criterion(logits, expected_tokens)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train() -> None:
    """Train the LSTM model and save the best checkpoint."""
    set_seed(SEED)
    dataset, vocab = load_dataset(DATASET_PATH)
    pad_idx = vocab["PAD"]
    vocab_size = len(vocab)

    training_loader, validation_loader = create_data_loaders(dataset)
    model = build_model(vocab_size, pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)

    best_validation_loss = float("inf")
    stalled_epochs = 0

    for epoch_index in range(NUM_EPOCHS):
        teacher_forcing_ratio = max(
            MIN_TEACHER_FORCING,
            1.0 - (epoch_index / NUM_EPOCHS) * 0.6,
        )
        training_loss = run_epoch(
            model,
            training_loader,
            criterion,
            optimizer,
            vocab_size,
            teacher_forcing_ratio,
            train_mode=True,
        )

        with torch.no_grad():
            validation_loss = run_epoch(
                model,
                validation_loader,
                criterion,
                optimizer,
                vocab_size,
                0.0,
                train_mode=False,
            )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch_index + 1}/{NUM_EPOCHS} | "
            f"train_loss={training_loss:.4f} | val_loss={validation_loss:.4f} | "
            f"teacher_forcing={teacher_forcing_ratio:.2f} | lr={current_lr:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stalled_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best LSTM model to {MODEL_PATH}")
        else:
            stalled_epochs += 1
            if stalled_epochs >= PATIENCE:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()
