""" Convert wav to lmfb (as numpy array)

Reference:
    https://github.com/mori97/librispeech-ctc/blob/master/src/preprocess.py
"""

import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import tensor2np


def main(args):
    with torch.no_grad():
        wav, sr = torchaudio.load(args.wav_path)
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
        print(lmfb)
        np.save(args.wav_path.replace(".wav", ".npy"), lmfb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=str)
    args = parser.parse_args()
    main(args)
