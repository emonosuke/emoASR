import argparse
import os

import pandas as pd


def split(args):
    df = pd.read_table(args.tsv_path)
    print(f"Data size: {len(df):d}")

    if args.shuffle:
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        print(f"Data shuffled")
    else:
        print(f"Data NOT shuffled")

    out_dir = os.path.splitext(args.tsv_path)[0]
    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.n_splits - 1):
        s_id = int((i / args.n_splits) * len(df))
        t_id = int(((i + 1) / args.n_splits) * len(df)) - 1
        df_part = df.loc[s_id:t_id]
        out_path = os.path.join(out_dir, f"part{(i+1):d}of{args.n_splits:d}.tsv")
        df_part.to_csv(out_path, index=False, sep="\t")
        print(f"Data[{s_id:d}:{t_id:d}] (size={t_id - s_id + 1}) saved to {out_path}")

    s_id = int(((args.n_splits - 1) / args.n_splits) * len(df))
    t_id = len(df) - 1
    df_part = df.loc[s_id:t_id]
    out_path = os.path.join(out_dir, f"part{args.n_splits:d}of{args.n_splits:d}.tsv")
    df_part.to_csv(out_path, index=False, sep="\t")
    print(f"Data[{s_id:d}:{t_id:d}] (size={t_id-s_id+1}) saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("-n_splits", type=int, required=True)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    split(args)
