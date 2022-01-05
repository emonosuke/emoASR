""" Align n-best hypotheses to train `electra-disc`, `pelectra-disc`
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from asr.metrics import compute_wer
from utils.converters import str2ints
from utils.paths import get_eval_path


def alignment(dfhyp, dfref, align_type="SID", len_min=1, len_max=256):
    id2ref = {}

    for row in dfref.itertuples():
        id2ref[row.utt_id] = str2ints(row.token_id)
    
    outs = []

    for row in tqdm(dfhyp.itertuples()):
        hyp_token_id = str2ints(row.token_id)
        ref_token_id = id2ref[row.utt_id]

        if len(hyp_token_id) < len_min or len(hyp_token_id) > len_max:
            continue

        _, wer_dict = compute_wer(hyp_token_id, ref_token_id)
        error_list = wer_dict["error_list"]

        align_list = []
        del_flag = False

        if align_type == "SI":
            align_list = [e for e in error_list if e != "D"]
        elif align_type == "SID":
            for e in error_list:
                if e == "D":
                    # pass `D` to left
                    if len(align_list) > 0 and align_list[-1] == "C":
                        align_list[-1] == "D"
                    else: # to right
                        del_flag = True
                else:
                    if del_flag and e == "C":
                        align_list.append("D")
                    else:
                        align_list.append(e)
                    del_flag = False

        assert len(hyp_token_id) == len(align_list)

        outs.append(
            (row.utt_id, row.score_asr, row.token_id, row.text, row.reftext, " ".join(align_list))
        )
    
    df = pd.DataFrame(
        outs, columns=["utt_id", "score_asr", "token_id", "text", "reftext", "error_label"]
    )

    return df

def main(args):
    dfhyp = pd.read_table(args.tsv_path)
    dfhyp = dfhyp.dropna()
    dfref = pd.read_table(get_eval_path(args.ref))

    df = alignment(dfhyp, dfref, args.align_type, len_min=args.len_min, len_max=args.len_max)

    df.to_csv(args.tsv_path.replace(".tsv", f"_{args.align_type}align.tsv"), sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("-ref", type=str, required=True)
    parser.add_argument("--align_type", choices=["SI", "SID"], default="SID")
    parser.add_argument("--len_min", type=int, default=1)
    parser.add_argument("--len_max", type=int, default=256)
    args = parser.parse_args()

    main(args)
