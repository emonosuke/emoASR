""" Convert wav to lmfb (as numpy array)
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import tensor2np


def save_feats(wav_path):
    with torch.no_grad():
        wav, sr = torchaudio.load(wav_path)
        assert sr == 16000
        wav *= 2 ** 15  # kaldi
        lmfb = torchaudio.compliance.kaldi.fbank(
            wav,
            window_type="hamming",
            htk_compat=True,
            sample_frequency=16000,
            num_mel_bins=80,
            use_energy=False,
        )
        lmfb = tensor2np(lmfb)

        npy_path = wav_path.replace(".wav", ".npy")
        np.save(npy_path, lmfb)

        print(f"{wav_path} -> {npy_path}")

        lmfb_sum = np.sum(lmfb, axis=0)
        lmfb_sqsum = np.sum(lmfb * lmfb, axis=0)
        num_frames = lmfb.shape[0]

    return lmfb_sum, lmfb_sqsum, num_frames


def main(args):
    if args.data_path.endswith(".tsv"):
        lmfb_sum_all, lmfb_sqsum_all = [], []
        num_frames_all = 0
        data = pd.read_table(args.data_path)
        for row in data.itertuples():
            lmfb_sum, lmfb_sqsum, num_frames = save_feats(row.wav_path)
            lmfb_sum_all.extend(lmfb_sum)
            lmfb_sqsum_all.extend(lmfb_sqsum)
            num_frames_all += num_frames
        norm_info = {}
        norm_info["lmfb_sum"] = lmfb_sum
        norm_info["lmfb_sqsum"] = lmfb_sqsum
        norm_info["num_frames"] = num_frames

        pickle_path = args.data_path.replace(".tsv", "_norm.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(norm_info, f)

    elif args.data_path.endswith(".wav"):
        lmfb_sum, lmfb_sqsum, num_frames = save_feats(args.data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    args = parser.parse_args()
    main(args)
