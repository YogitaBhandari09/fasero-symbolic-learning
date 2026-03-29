"""Generate symbolic function and Taylor-series pairs with SymPy."""

import argparse
import json
from pathlib import Path
import random
import sys

import sympy as sp
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "taylor_dataset.json"
x = sp.symbols("x")

BASE_FUNCTIONS = [
    sp.sin(x),
    sp.cos(x),
    sp.exp(x),
    sp.log(1 + x),
    x,
    x**2,
    x**3,
]


def parse_args():
    """Read command-line arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate symbolic Taylor-series training data.")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("--series-order", type=int, default=5, help="Taylor expansion order.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def set_seed(seed):
    """Set the random seed used for symbolic sampling."""
    random.seed(seed)


def generate_function():
    """Sample and lightly compose a symbolic function."""
    sampled_expression = random.choice(BASE_FUNCTIONS)

    if random.random() < 0.5:
        sampled_expression = sampled_expression + random.choice(BASE_FUNCTIONS)

    if random.random() < 0.3:
        sampled_expression = sampled_expression - random.choice(BASE_FUNCTIONS)

    return sp.simplify(sampled_expression)


def taylor_expand(function_expr, series_order):
    """Compute a truncated Taylor expansion around x = 0."""
    series = sp.series(function_expr, x, 0, series_order)
    return sp.simplify(series.removeO())


def main():
    """Generate the raw dataset and write it to disk."""
    args = parse_args()
    set_seed(args.seed)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_rows = []

    for _ in tqdm(range(args.num_samples), desc="Generating dataset"):
        function_expr = generate_function()
        taylor_expr = taylor_expand(function_expr, args.series_order)
        dataset_rows.append({"function": str(function_expr), "taylor": str(taylor_expr)})

    with RAW_DATA_PATH.open("w", encoding="utf-8") as output_file:
        json.dump(dataset_rows, output_file, indent=2)

    print(f"Dataset generated with {len(dataset_rows)} samples at {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
