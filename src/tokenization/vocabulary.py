from collections import Counter

class Vocabulary:

    def __init__(self):
        self.token_to_id = {"PAD":0,"START":1,"END":2}
        self.id_to_token = {0:"PAD",1:"START",2:"END"}

    def build(self,token_lists):

        counter = Counter()

        for tokens in token_lists:
            counter.update(tokens)

        for token in counter:

            if token not in self.token_to_id:

                idx=len(self.token_to_id)

                self.token_to_id[token]=idx
                self.id_to_token[idx]=token

    def encode(self,tokens):

        return [self.token_to_id[t] for t in tokens]

    def decode(self,ids):

        return [self.id_to_token[i] for i in ids]