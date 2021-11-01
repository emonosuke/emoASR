"""
Add phone mapping to tsv (as `phone_token_id`, `phone_text`)
"""

import argparse
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import ints2str
from utils.vocab import Vocab


def main(args):
    word2phone = {}
    with open(args.lexicon, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(r"[\s]+", " ", line.strip())  # Remove successive spaces
            word = line.split(" ")[0]
            word = word.split("+")[0]  # for CSJ
            word = word.lower()  # for Librispeech
            phone_seq = " ".join(line.split(" ")[1:])
            word2phone[word] = phone_seq
    vocab = Vocab(args.vocab)

    if args.input.endswith(".tsv"):
        tsv_path = args.input
        df = pd.read_table(tsv_path)
        df = df.dropna(subset=["utt_id", "token_id", "text"])

        phone_texts = []
        phone_token_ids = []

        for row in tqdm(df.itertuples()):
            # print("text:", row.text)
            # print("token_id:", row.token_id)
            words = row.text.split(" ")
            phones = []
            for w in words:
                if w in word2phone:
                    phones += word2phone[w].split()
                else:
                    phones += [args.unk]
            phone_text = " ".join(phones)
            phone_token_id = ints2str(vocab.tokens2ids(phones))

            # print("phone_text:", phone_text)
            # print("phone_token_id:", phone_token_id)

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
            df.to_csv(tsv_path.replace(".tsv", "_p2w.tsv"), sep="\t", index=False)
        else:
            df.to_csv(args.out, sep="\t", index=False)
    else:
        words = args.input.split(" ")
        phones = []
        for w in words:
            if w in word2phone:
                phones += word2phone[w].split()
            else:
                phones += [args.unk]
        phone_text = " ".join(phones)
        phone_token_id = ints2str(vocab.tokens2ids(phones))

        print(f"text: {phone_text}")
        print(f"token_id: {phone_token_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)  # tsv_path or text
    parser.add_argument("-lexicon", type=str, required=True)
    parser.add_argument("-vocab", type=str, required=True)
    parser.add_argument("--unk", type=str, default="NSN")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--cols", type=str, default=None
    )  # utt_id,token_id,text,phone_token_id,phone_text
    args = parser.parse_args()
    main(args)
