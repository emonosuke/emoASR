""" Shallow Fusion with grid search parameter search
"""
import argparse
import logging
import multiprocessing
import os
import sys

import numpy as np

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from asr.test_asr import test_main
from utils.paths import get_eval_path, get_results_dir

EPS = 1e-5


def main(args):
    log_dir = get_results_dir(args.conf)
    data_path = get_eval_path(args.data)
    data_tag = (
        args.data
        if args.data_tag == "test" and data_path != args.data
        else args.data_tag
    )
    log_file = (
        f"test_fusion_grid_{data_tag}_ctc{args.decode_ctc_weight}_ep{args.ep}.log"
    )

    logging.basicConfig(
        filename=os.path.join(log_dir, log_file),
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    # grid search
    lm_weight_cands = np.arange(args.lm_min, args.lm_max + EPS, args.lm_step)
    len_weight_cands = np.arange(args.len_min, args.len_max + EPS, args.len_step)
    pool = multiprocessing.Pool(len(lm_weight_cands) * len(len_weight_cands))

    func_args = []

    for lm_weight in lm_weight_cands:
        for len_weight in len_weight_cands:
            func_args.append((args, lm_weight, len_weight))

    results = pool.starmap(test_main, func_args)

    lm_weight_min = 0
    len_weight_min = 0
    wer_min = 100
    wer_info_min = ""

    for lm_weight, len_weight, wer, wer_info in results:
        logging.info(
            f"lm_weight: {lm_weight:.3f} len_weight: {len_weight:.3f} - {wer_info}"
        )
        if wer < wer_min:
            lm_weight_min = lm_weight
            len_weight_min = len_weight
            wer_min = wer
            wer_info_min = wer_info

    logging.info("***** best WER:")
    logging.info(
        f"lm_weight: {lm_weight_min:.3f} len_weight: {len_weight_min:.3f} - {wer_info_min}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--decode_ctc_weight", type=float, default=0)
    #
    parser.add_argument("--lm_min", type=float, default=0)
    parser.add_argument("--lm_max", type=float, default=1)
    parser.add_argument("--lm_step", type=float, default=0.1)
    parser.add_argument("--len_min", type=float, default=0)
    parser.add_argument("--len_max", type=float, default=5)
    parser.add_argument("--len_step", type=float, default=1)
    parser.add_argument("--lm_conf", type=str, default=None)
    parser.add_argument("--lm_ep", type=str, default=None)
    parser.add_argument("--lm_tag", type=str, default=None)
    args = parser.parse_args()

    # set unused attributes
    args.cpu = True
    args.nbest = False
    args.debug = False
    args.utt_id = None
    args.runtime = False
    main(args)
