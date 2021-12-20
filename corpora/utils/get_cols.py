import argparse

import pandas as pd


def main(args):
    data = pd.read_table(args.tsv_path)
    print(f"Read tsv ({len(data)} samples)")

    columns = [column for column in args.cols.split(",")]
    data = data[columns]
    data.to_csv(args.out, index=False, header=(not args.no_header), sep="\t")
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("-cols", type=str, required=True)
    parser.add_argument("-out", type=str, required=True)
    parser.add_argument("--no_header", action="store_true")
    args = parser.parse_args()
    main(args)
