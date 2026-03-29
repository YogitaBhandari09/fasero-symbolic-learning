"""Simple regex-based tokenizer for symbolic math expressions."""

import re


TOKEN_PATTERN = r"\d+|[a-zA-Z]+|\*\*|\S"


def tokenize(expression):
    """Split a symbolic expression into model-friendly tokens."""
    return re.findall(TOKEN_PATTERN, expression)


if __name__ == "__main__":
    example_expression = "x - x**3/6"
    print(tokenize(example_expression))
