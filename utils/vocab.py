import torch
from torch.nn.utils.rnn import pad_sequence


class Vocab:
    def __init__(self, vocab_path: str):
        with open(vocab_path) as f:
            lines = [line.strip() for line in f]

        i2t = {}
        t2i = {}
        for line in lines:
            token, idx = tuple(line.split())
            i2t[int(idx)] = token
            t2i[token] = int(idx)
        self.i2t = i2t
        self.t2i = t2i

        self.unk_id = t2i["<unk>"]

    def id2token(self, idx):
        return self.i2t[idx]

    def ids2tokens(self, ids):
        return [self.id2token(i) for i in ids]

    def id2words(self, ids):
        return self.subwords_to_words(self.ids2tokens(ids))

    def ids2text(self, ids):
        return " ".join(self.subwords_to_words(self.ids2tokens(ids)))

    def token2id(self, word):
        if word in self.t2i:
            return self.t2i[word]
        return self.unk_id

    def tokens2ids(self, words):
        return [self.token2id(w) for w in words]

    def is_subword(self, idx):
        subword = self.id2word(idx)
        return subword[0] != "_" and subword[0] != "<"

    def subwords_to_words(self, subwords):
        """ assume BPE style as in https://github.com/google/sentencepiece
        """
        tmp = ""
        words = []
        for subword in subwords:
            if (
                subword[0] == "▁" or subword[0] == "<" or (tmp and tmp[-1] == ">")
            ):  # appear new word
                if tmp != "":
                    words.append(tmp)
                    tmp = ""

                tmp += subword[1:] if subword[0] == "▁" else subword
            else:
                tmp += subword

        if tmp != "":
            words.append(tmp)
        return words
