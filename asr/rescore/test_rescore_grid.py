""" Rescoring with grid search parameter search
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from asr.metrics import compute_wers_df
from lm.modeling.lm import LM
from utils.configure import load_config
from utils.converters import str2ints, tensor2np
from utils.paths import get_eval_path, get_model_path, rel_to_abs_path
from utils.vocab import Vocab

BATCH_SIZE = 100
EPS = 1e-5


def score_lm(df, model, device, mask_id=None, vocab=None, num_samples=-1):
    ys, ylens, score_lms_all = [], [], []

    utt_id = None
    cnt_utts = 0

    for i, row in enumerate(df.itertuples()):
        if row.utt_id != utt_id:
            cnt_utts += 1
            utt_id = row.utt_id
        if num_samples > 0 and (cnt_utts + 1) > num_samples:
            return

        y = str2ints(row.token_id)
        ys.append(torch.tensor(y))
        ylens.append(len(y))

        if len(ys) < BATCH_SIZE and i != (len(df) - 1):
            continue

        ys_pad = pad_sequence(ys, batch_first=True).to(device)
        ylens = torch.tensor(ylens).to(device)
        
        score_lms = model.score(ys_pad, ylens, batch_size=BATCH_SIZE)

        if vocab is not None:  # debug mode
            for y, score_lm in zip(ys, score_lms):
                logging.debug(
                    f"{' '.join(vocab.ids2words(tensor2np(y)))}: {score_lm:.3f}"
                )

        score_lms_all.extend(score_lms)
        ys, ylens = [], []

    df["score_lm"] = score_lms_all
    return df


def rescore(df, dfref, lm_weight, len_weight):
    df["ylen"] = df["token_id"].apply(lambda s: len(s.split()))
    df["score"] = df["score_asr"] + lm_weight * df["score_lm"] + len_weight * df["ylen"]

    df_best = df.loc[df.groupby("utt_id")["score"].idxmax(), :]
    df_best = df_best[["utt_id", "text", "token_id", "score_asr"]]

    wer, wer_dict = compute_wers_df(df_best, dfref)
    return wer, wer_dict, df_best


def main(args):
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lm_params = load_config(args.lm_conf)
    lm_tag = lm_params.lm_type if args.lm_tag is None else args.lm_tag
    if args.debug or args.runtime:
        logging.basicConfig(
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.DEBUG,
        )
    else:
        log_path = args.tsv_path.replace(".tsv", f"_{lm_tag}.log")
        logging.basicConfig(
            filename=log_path,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.INFO,
        )
    
    df = pd.read_table(args.tsv_path)
    df = df.dropna()
    dfref = pd.read_table(get_eval_path(args.ref))

    # LM
    lm_path = get_model_path(args.lm_conf, args.lm_ep)
    lm_params = load_config(args.lm_conf)
    lm = LM(lm_params, phase="test")
    lm.load_state_dict(torch.load(lm_path, map_location=device))
    logging.info(f"LM: {lm_path}")
    lm.to(device)
    lm.eval()

    mask_id = lm_params.mask_id if hasattr(lm_params, "mask_id") else None
    vocab = Vocab(rel_to_abs_path(lm_params.vocab_path)) if args.debug else None

    if args.runtime:
        torch.set_num_threads(1)

        global BATCH_SIZE
        BATCH_SIZE = 1
        runtimes = []

        for j in range(args.runtime_num_repeats):
            start_time = time.time()

            score_lm(df, lm, device, mask_id=mask_id, num_samples=args.runtime_num_samples)
            
            runtime = time.time() - start_time
            runtime /= args.runtime_num_samples
            logging.info(f"Run {(j+1):d} runtime: {runtime:.5f}sec / utt")
            runtimes.append(runtime)

        logging.info(f"Average runtime {np.mean(runtimes):.5f}sec on {device.type}")
        return
    
    scored_tsv_path = args.tsv_path.replace(".tsv", f"_{lm_tag}.tsv")

    # calculate `score_lm`
    if not os.path.exists(scored_tsv_path):
        df = score_lm(df, lm, device, mask_id=mask_id, vocab=vocab)
        df.to_csv(scored_tsv_path, sep="\t", index=False)
    else:
        logging.info(f"load score_lm: {scored_tsv_path}")
        df = pd.read_table(scored_tsv_path)

    # grid search
    lm_weight_cands = np.arange(args.lm_min, args.lm_max + EPS, args.lm_step)
    len_weight_cands = np.arange(args.len_min, args.len_max + EPS, args.len_step)

    lm_weight_best = 0
    len_weight_best = 0
    df_best = None
    wer_min = 100

    for lm_weight in lm_weight_cands:
        for len_weight in len_weight_cands:
            wer, wer_dict, df_result = rescore(df, dfref, lm_weight, len_weight)

            wer_info = f"WER: {wer:.2f} [D={wer_dict['n_del']:d}, S={wer_dict['n_sub']:d}, I={wer_dict['n_ins']:d}, N={wer_dict['n_ref']:d}]"
            logging.info(
                f"lm_weight: {lm_weight:.3f} len_weight: {len_weight:.3f} - {wer_info}"
            )

            if wer < wer_min:
                wer_min = wer
                lm_weight_best = lm_weight
                len_weight_best = len_weight
                df_best = df_result

    best_tsv_path = scored_tsv_path.replace(
        ".tsv", f"_lm{lm_weight_best:.2f}_len{len_weight_best:.2f}.tsv"
    )
    logging.info(
        f"best lm_weight: {lm_weight_best:.3f} len_weight: {len_weight_best:.3f}"
    )
    if df_best is not None:
        df_best.to_csv(best_tsv_path, sep="\t", index=False)
    logging.info(f"best WER: {wer_min:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)  # nbest
    parser.add_argument("-ref", type=str, required=True)  # tsv_path for reference
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--runtime_num_samples", type=int, default=20)
    parser.add_argument("--runtime_num_repeats", type=int, default=5)
    #
    parser.add_argument("-lm_conf", type=str, required=True)
    parser.add_argument("-lm_ep", type=str, required=True)
    parser.add_argument("--lm_tag", type=str, default=None)
    parser.add_argument("--lm_min", type=float, default=0)
    parser.add_argument("--lm_max", type=float, default=1)
    parser.add_argument("--lm_step", type=float, default=0.1)
    parser.add_argument("--len_min", type=float, default=0)
    parser.add_argument("--len_max", type=float, default=5)
    parser.add_argument("--len_step", type=float, default=1)
    args = parser.parse_args()
    main(args)
