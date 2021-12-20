import argparse

import pandas as pd

IGNORE_TEXT = "ignore_time_segment_in_scoring"


def main(args):
    df = pd.read_table(args.tsv_path)
    df_rm = df[df["text"] != IGNORE_TEXT]
    print(f"remove {IGNORE_TEXT} in {args.tsv_path}: {len(df):d} -> {len(df_rm):d}")
    df_rm.to_csv(args.tsv_path, index=False, sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    args = parser.parse_args()
    main(args)
