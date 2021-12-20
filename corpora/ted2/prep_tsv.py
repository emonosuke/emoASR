import argparse
import os
import sys

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

import numpy as np
import pandas as pd
from utils.converters import str2ints


def get_xlen(npy_path):
    x = np.load(npy_path)
    return len(x)


def get_ylen(token_id):
    return len(str2ints(token_id))


def main(args):
    df = pd.read_table(args.tsv_path)
    if "wav_path" in df:
        df["wav_path"] = df["wav_path"].str.replace(".wav", "_norm.npy", regex=False)
        df["wav_path"] = "/n/work1/futami/emoASR/corpora/" + df["wav_path"]
        df = df.rename(columns={"wav_path": "feat_path"})
    if "xlen" not in df:
        df["xlen"] = df["feat_path"].map(get_xlen)
    if "ylen" not in df:
        df["ylen"] = df["token_id"].map(get_ylen)

    df.to_csv(args.tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    args = parser.parse_args()
    main(args)
