import argparse
import os

import pandas as pd


def sort_data(data, task):
    if task == "ASR":
        # TODO: get xlen if does not exist
        data_sorted = data.sort_values(["xlen"])

    elif task == "P2W":
        if "plen" not in data:
            data["plen"] = data["phone_token_id"].str.split().str.len()
        data_sorted = data.sort_values(["plen"])

    return data_sorted


def main(args):
    if os.path.isdir(args.tsv_path):
        for tsv_file in os.listdir(args.tsv_path):
            tsv_file_path = os.path.join(args.tsv_path, tsv_file)
            data = pd.read_table(tsv_file_path)
            data_sorted = sort_data(data, args.task)
            save_path = f"{os.path.splitext(tsv_file_path)[0]}_sorted.tsv"
            # NOTE: inplace
            data_sorted.to_csv(tsv_file_path, sep="\t", index=False)
            print(f"sorted data saved to: {tsv_file_path}")
    else:
        data = pd.read_table(args.tsv_path)
        data_sorted = sort_data(data, args.task)
        save_path = f"{os.path.splitext(args.tsv_path)[0]}_sorted.tsv"
        data_sorted.to_csv(save_path, sep="\t", index=False)
        print(f"sorted data saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("--task", choices=["ASR", "P2W"], default="ASR")
    args = parser.parse_args()
    main(args)
