import re

def tokenize(expr):

    tokens = re.findall(r'\d+|[a-zA-Z]+|\*\*|\S', expr)

    return tokens


if __name__ == "__main__":

    expr = "x - x**3/6"

    print(tokenize(expr))
