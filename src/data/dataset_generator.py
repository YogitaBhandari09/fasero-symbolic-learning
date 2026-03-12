import sympy as sp
import random
import json
from tqdm import tqdm

x = sp.symbols('x')

base_functions = [
    sp.sin(x),
    sp.cos(x),
    sp.exp(x),
    sp.log(1+x),
    x,
    x**2,
    x**3
]

def generate_function():
    f = random.choice(base_functions)

    if random.random() < 0.5:
        g = random.choice(base_functions)
        f = f + g

    return sp.simplify(f)

def taylor_expand(f):
    series = sp.series(f, x, 0, 5)
    series = series.removeO()
    return sp.simplify(series)

dataset = []

for _ in tqdm(range(5000)):
    f = generate_function()
    taylor = taylor_expand(f)

    dataset.append({
        "function": str(f),
        "taylor": str(taylor)
    })

with open("data/raw/taylor_dataset.json","w") as f:
    json.dump(dataset,f,indent=2)

print("Dataset generated!")
