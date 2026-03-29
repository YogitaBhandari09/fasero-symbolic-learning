import math

import torch
import torch.nn as nn


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_heads=8,
        num_layers=3,
        forward_expansion=512,
        dropout=0.1,
        max_len=30,
        pad_idx=0,
    ):
        super().__init__()

        self.embed_size = embed_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "pos_encoding",
            self.create_positional_encoding(max_len, embed_size),
            persistent=False,
        )

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=forward_expansion,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)

    @staticmethod
    def create_positional_encoding(max_len, embed_size):
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        return pe.unsqueeze(0)

    def encode_tokens(self, tokens):
        seq_len = tokens.shape[1]
        embedding = self.embedding(tokens) * math.sqrt(self.embed_size)
        positional = self.pos_encoding[:, :seq_len]
        return self.dropout(embedding + positional)

    def make_padding_mask(self, tokens):
        return tokens.eq(self.pad_idx)

    def forward(self, src, trg):
        _, trg_len = trg.shape

        src_embed = self.encode_tokens(src)
        trg_embed = self.encode_tokens(trg)

        src_padding_mask = self.make_padding_mask(src)
        trg_padding_mask = self.make_padding_mask(trg)
        tgt_mask = torch.triu(
            torch.ones(trg_len, trg_len, device=src.device, dtype=torch.bool),
            diagonal=1,
        )

        out = self.transformer(
            src_embed,
            trg_embed,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )

        return self.fc_out(out)
