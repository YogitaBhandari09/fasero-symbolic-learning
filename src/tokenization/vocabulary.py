"""Vocabulary helper for converting symbolic tokens to integer ids."""

from collections import Counter


class Vocabulary:
    """Store token-id mappings used during preprocessing and decoding."""

    def __init__(self):
        self.token_to_id = {"PAD": 0, "START": 1, "END": 2}
        self.id_to_token = {0: "PAD", 1: "START", 2: "END"}

    def build(self, token_lists):
        """Expand the vocabulary from an iterable of token sequences."""
        token_counter = Counter()

        for tokens in token_lists:
            token_counter.update(tokens)

        for token in token_counter:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token

    def encode(self, tokens):
        """Map a token sequence to integer ids."""
        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids):
        """Map integer ids back to tokens."""
        return [self.id_to_token[token_id] for token_id in token_ids]
