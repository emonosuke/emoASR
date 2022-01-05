""" Measures soft label accuracy
"""

import argparse
import os
import pickle
import sys

import pandas as pd
from tqdm import tqdm

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from utils.converters import str2ints
from utils.paths import get_eval_path
from utils.vocab import Vocab


def accuracy(labels, dfref, vocab=None):
    id2ref = {}
    cnt, cntacc1, cntacck = 0, 0, 0

    for row in dfref.itertuples():
        id2ref[row.utt_id] = str2ints(row.token_id)
        # assert row.utt_id in labels.keys()

    for utt_id, label in tqdm(labels.items()):
        ref_token_id = id2ref[utt_id]
        cnt += len(label)

        if vocab is not None:
            print(f"# utt_id: {utt_id}")

            ref_text = vocab.ids2tokens(ref_token_id)
            for i, vps in enumerate(label):
                # mask i-th token
                ref_text_masked = ref_text.copy()
                ref_text_masked[i] = "<mask>"
                print(" ".join(ref_text_masked))

                for v, p in vps:
                    print(f"{vocab.id2token(v)}: {p:.2f}", end=" ")
                print()

        for i, vps in enumerate(label):
            v1, _ = vps[0]
            cntacc1 += int(v1 == ref_token_id[i])

            for v, _ in vps:
                cntacck += int(v == ref_token_id[i])

    acc1 = (cntacc1 / cnt) * 100
    acck = (cntacck / cnt) * 100

    return acc1, acck, cnt


def main(args):
    with open(args.pkl_path, "rb") as f:
        labels = pickle.load(f)
    print("pickle loaded")

    tsv_path = get_eval_path(args.ref)
    dfref = pd.read_table(tsv_path)

    if args.vocab is not None:
        vocab = Vocab(args.vocab)
    else:
        vocab = None

    acc1, acck, cnt = accuracy(labels, dfref, vocab=vocab)

    print(f"{cnt:d} tokens")
    print(f"Accuracy top1: {acc1:.3f} topk: {acck:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_path", type=str)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("--vocab", type=str)  # debug
    args = parser.parse_args()

    main(args)
