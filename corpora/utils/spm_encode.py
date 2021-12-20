import argparse
import os
import sys

import pandas as pd

# https://github.com/google/sentencepiece
import sentencepiece as spm
from tqdm import tqdm

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import ints2str
from utils.vocab import Vocab


def main(args):
    df = pd.read_table(args.data)
    df = df.dropna()

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)
    vocab = Vocab(args.vocab)

    token_ids = []

    for row in tqdm(df.itertuples()):
        tokens = sp.EncodeAsPieces(row.text)
        token_id = vocab.tokens2ids(tokens)
        token_ids.append(ints2str(token_id))

    df["token_id"] = token_ids

    if args.out is None:
        # overwrite
        df.to_csv(args.data, sep="\t", index=False)
    else:
        df.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)  # .tsv
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-vocab", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    main(args)
