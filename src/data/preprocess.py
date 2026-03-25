import json
from src.tokenization.tokenizer import tokenize
from src.tokenization.vocabulary import Vocabulary

MAX_LEN = 30  # keep small for now

def pad_sequence(seq, max_len):
    return seq + [0]*(max_len - len(seq))


def main():

    with open("data/raw/taylor_dataset.json") as f:
        data = json.load(f)

    input_tokens = []
    target_tokens = []

    # tokenize
    for sample in data:
        inp = tokenize(sample["function"])
        tgt = tokenize(sample["taylor"])

        input_tokens.append(inp)
        target_tokens.append(tgt)

    # build vocab
    vocab = Vocabulary()
    vocab.build(input_tokens + target_tokens)

    # encode
    encoded_inputs = []
    encoded_targets = []

    for inp, tgt in zip(input_tokens, target_tokens):

        inp_ids = [1] + vocab.encode(inp) + [2]  # START, END
        tgt_ids = [1] + vocab.encode(tgt) + [2]

        inp_ids = pad_sequence(inp_ids, MAX_LEN)
        tgt_ids = pad_sequence(tgt_ids, MAX_LEN)

        encoded_inputs.append(inp_ids)
        encoded_targets.append(tgt_ids)

    # save processed data
    processed = {
        "inputs": encoded_inputs,
        "targets": encoded_targets,
        "vocab": vocab.token_to_id
    }

    with open("data/processed/dataset.json","w") as f:
        json.dump(processed,f)

    print("Preprocessing done!")


if __name__ == "__main__":
    main()
