SPECIAL_TOKEN_IDS = {0, 1, 2}


def normalize_token_sequence(tokens, special_token_ids=None):
    ids_to_skip = SPECIAL_TOKEN_IDS if special_token_ids is None else set(special_token_ids)
    normalized = []

    for token in tokens:
        token_id = int(token)
        if token_id in ids_to_skip:
            continue
        normalized.append(token_id)

    return normalized


def token_accuracy(pred, target, special_token_ids=None):
    pred_tokens = normalize_token_sequence(pred, special_token_ids)
    target_tokens = normalize_token_sequence(target, special_token_ids)

    total = max(len(target_tokens), 1)
    correct = sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))

    return correct / total


def exact_match(pred, target, special_token_ids=None):
    pred_tokens = normalize_token_sequence(pred, special_token_ids)
    target_tokens = normalize_token_sequence(target, special_token_ids)
    return int(pred_tokens == target_tokens)
