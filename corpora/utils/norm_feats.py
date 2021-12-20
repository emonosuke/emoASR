import argparse
import pickle

import numpy as np
import pandas as pd


def save_feats(npy_path, mean, std):
    out_npy_path = npy_path.replace(".npy", "_norm.npy")
    x = np.load(npy_path)
    # print(x)
    x_norm = (x - mean) / std
    # print(x_norm)
    print(out_npy_path)
    np.save(out_npy_path, x_norm)


def main(args):
    norm_paths = args.norm_path.split(",")

    lmfb_sum, lmfb_sqsum = None, None
    num_frames = 0
    for norm_path in norm_paths:
        with open(norm_path, "rb") as f:
            norm_info = pickle.load(f)
        if lmfb_sum is None:
            lmfb_sum = norm_info["lmfb_sum"]
            lmfb_sqsum = norm_info["lmfb_sqsum"]
        else:
            lmfb_sum += norm_info["lmfb_sum"]
            lmfb_sqsum += norm_info["lmfb_sqsum"]
        num_frames += norm_info["num_frames"]

    mean = lmfb_sum / num_frames
    var = lmfb_sqsum / num_frames - (mean * mean)
    std = np.sqrt(var)

    if args.data_path.endswith(".tsv"):
        df = pd.read_table(args.data_path)
        for row in df.itertuples():
            npy_path = row.wav_path.replace(".wav", ".npy")
            save_feats(npy_path, mean, std)
    elif args.data_path.endswith(".npy"):
        save_feats(args.data_path, mean, std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("norm_path", type=str)
    args = parser.parse_args()
    main(args)
