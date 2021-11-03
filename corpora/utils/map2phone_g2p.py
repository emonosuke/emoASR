"""
Add phone mapping to tsv (as `phone_token_id`, `phone_text`) with g2p (openjtalk)
"""

import argparse
import os
import sys

import pandas as pd
import pyopenjtalk
from tqdm import tqdm

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import ints2str
from utils.vocab import Vocab


def build_vocab(df, vocab_path):
    print(f"building vocab ...")

    vocab_dict = {"<unk>": 1, "<eos>": 2, "<pad>": 3}
    vocab_set = []

    for row in tqdm(df.itertuples()):
        text = row.text.replace(" ", "")  # remove spaces

        phones = pyopenjtalk.g2p(text, join=False)
        # remove pause
        phones = [phone for phone in phones if phone != "pau"]

        for phone in phones:
            if phone not in vocab_set:
                vocab_set.append(phone)

    # alphabetical order
    vocab_set.sort()

    wlines = []
    for v in vocab_set:
        index = len(vocab_dict) + 1
        vocab_dict[v] = index

    for v, index in vocab_dict.items():
        wlines.append(f"{v} {index:d}\n")

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.writelines(wlines)

    print(f"vocabulary saved to {vocab_path}")

    return Vocab(vocab_path)


def main(args):
    df = pd.read_table(args.tsv_path)
    df = df.dropna(subset=["utt_id", "token_id", "text"])

    if not os.path.exists(args.vocab):
        vocab = build_vocab(df, args.vocab)
    else:
        vocab = Vocab(args.vocab)
        print(f"load vocab: {args.vocab}")

    phone_texts = []
    phone_token_ids = []

    for row in tqdm(df.itertuples()):
        text = row.text.replace(" ", "")  # remove spaces
        phones = pyopenjtalk.g2p(text, join=False)
        phone_text = " ".join(phones)
        phone_token_id = ints2str(vocab.tokens2ids(phones))

        phone_texts.append(phone_text)
        phone_token_ids.append(phone_token_id)

    df["phone_text"] = phone_texts
    df["phone_token_id"] = phone_token_ids

    if args.cols is not None:
        columns = [column for column in args.cols.split(",")]
        assert (
            ("utt_id" in columns)
            and ("phone_text" in columns)
            and ("phone_token_id" in columns)
        )
        df = df[columns]

    if args.out is None:
        df.to_csv(args.tsv_path.replace(".tsv", "_p2w.tsv"), sep="\t", index=False)
    else:
        df.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("-vocab", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--cols", type=str, default=None
    )  # utt_id,token_id,text,phone_token_id,phone_text
    args = parser.parse_args()
    main(args)
