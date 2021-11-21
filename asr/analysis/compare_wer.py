import argparse
import os
import sys

import numpy as np
import pandas as pd

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from asr.metrics import compute_wer
from utils.paths import get_eval_path


def main(args):
    dfhyp1 = pd.read_table(args.hyp1, comment="#")
    dfhyp2 = pd.read_table(args.hyp2, comment="#")
    dfref = pd.read_table(get_eval_path(args.ref), comment="#")

    id2hyp1, id2hyp2 = {}, {}
    cnt_na1, cnt_na2 = 0, 0

    for rowhyp1 in dfhyp1.itertuples():
        if pd.isna(rowhyp1.text):
            id2hyp1[rowhyp1.utt_id] = []
            cnt_na1 += 1
        else:
            id2hyp1[rowhyp1.utt_id] = rowhyp1.text.split()
    for rowhyp2 in dfhyp2.itertuples():
        if pd.isna(rowhyp2.text):
            id2hyp2[rowhyp2.utt_id] = []
            cnt_na2 += 1
        else:
            id2hyp2[rowhyp2.utt_id] = rowhyp2.text.split()

    for rowref in dfref.itertuples():
        hyp1 = id2hyp1[rowref.utt_id] if rowref.utt_id in id2hyp1 else ["<dummy>"]
        hyp2 = id2hyp2[rowref.utt_id] if rowref.utt_id in id2hyp2 else ["<dummy>"]
        ref = rowref.text.split()

        _, wer_dict1 = compute_wer(hyp1, ref)
        _, wer_dict2 = compute_wer(hyp2, ref)
        n_err1 = wer_dict1["n_del"] + wer_dict1["n_sub"] + wer_dict1["n_ins"]
        n_err2 = wer_dict2["n_del"] + wer_dict2["n_sub"] + wer_dict2["n_ins"]
        n_err_diff = abs(n_err1 - n_err2)

        if n_err1 == n_err2:
            neq = "="
        elif n_err1 < n_err2:
            neq = "<"
        else:
            neq = ">"

        if (
            (args.filter is None or args.filter == neq)
            and len(ref) <= args.max_len
            and n_err_diff >= args.min_diff
        ):
            print(f"utt_id: {rowref.utt_id}", flush=True)
            print(
                f"hyp1[D={wer_dict1['n_del']} S={wer_dict1['n_sub']} I={wer_dict1['n_ins']}] {neq} hyp2[D={wer_dict2['n_del']} S={wer_dict2['n_sub']} I={wer_dict2['n_ins']}]"
            )
            print(f"hyp1: {' '.join(hyp1)}")
            print(f"hyp2: {' '.join(hyp2)}")
            print(f"ref : {' '.join(ref)}")
            print("==========")

    print(f"cannot decode: hyp1: {cnt_na1:d}, hyp2: {cnt_na2:d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-hyp1", type=str)
    parser.add_argument("-hyp2", type=str)
    parser.add_argument("-ref", type=str)
    parser.add_argument("--filter", type=str, choices=["<", ">", "=", ""], default=None)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--min_diff", type=int, default=0)
    args = parser.parse_args()
    main(args)
