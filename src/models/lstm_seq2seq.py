import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx=0, num_layers=2, dropout=0.2):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_idx=0, num_layers=2, dropout=0.2):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        hidden, cell = self.encoder(src)
        input_token = trg[:, 0]

        for timestep in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, timestep] = output

            top1 = output.argmax(1)
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = trg[:, timestep] if use_teacher else top1

        return outputs
