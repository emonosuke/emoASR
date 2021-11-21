""" test ASR
"""
import argparse
import logging
import os
import socket
import sys
import time

import git
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.average_checkpoints import model_average
from utils.configure import load_config
from utils.converters import ints2str, strip_eos
from utils.log import insert_comment
from utils.paths import get_eval_path, get_model_path, get_results_dir, rel_to_abs_path
from utils.vocab import Vocab

from asr.datasets import ASRDataset
from asr.metrics import compute_wers_df
from asr.modeling.asr import ASR

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# TODO: shallow fusion and rescoring


def test_step(
    model, data, beam_width, len_weight, decode_ctc_weight, lm, lm_weight, device
):
    utt_id = data["utt_ids"][0]
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    reftext = data["texts"][0]

    hyps, scores, _, _ = model.decode(
        xs, xlens, beam_width, len_weight, decode_ctc_weight, lm=lm, lm_weight=lm_weight
    )
    return utt_id, hyps, scores, reftext


def test(
    model,
    dataloader,
    vocab,
    beam_width,
    len_weight,
    decode_ctc_weight,
    lm,
    lm_weight,
    device,
    eos_id=2,
    num_samples=-1,
    sample_utt_id=None,
):
    rows = []  # utt_id, token_id, text, reftext

    for i, data in enumerate(dataloader):
        if num_samples > 0 and (i + 1) > num_samples:
            return rows

        if sample_utt_id is not None and sample_utt_id != data["utt_ids"][0]:
            continue

        utt_id, hyps, scores, reftext = test_step(
            model,
            data,
            beam_width,
            len_weight,
            decode_ctc_weight,
            lm,
            lm_weight,
            device,
        )
        text = ""

        # FIXME: len(hyps[0]) < 1 ?
        if len(hyps) < 1:
            token_id = None
            text = ""
            logging.warning(f"cannot decode {utt_id}")
        else:
            # NOTE: strip <eos> (or <sos>) here
            token_id = ints2str(strip_eos(hyps[0], eos_id))
            text = vocab.ids2text(strip_eos(hyps[0], eos_id))

        rows.append([utt_id, token_id, text, reftext])

        logging.debug(f"{utt_id}({(i+1):d}/{len(dataloader):d}): {text}")

    return rows


def main(args):
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        # make sure all operations are done on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = load_config(args.conf)

    beam_width = args.beam_width if args.beam_width is not None else params.beam_width
    len_weight = args.len_weight if args.len_weight is not None else params.len_weight
    decode_ctc_weight = (
        args.decode_ctc_weight
        if args.decode_ctc_weight is not None
        else params.decode_ctc_weight
    )
    lm_weight = args.lm_weight if args.lm_weight > 0 else params.lm_weight

    if args.debug:
        logging.basicConfig(
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.INFO,
        )

    logging.info(f"***** {' '.join(sys.argv)}")
    logging.info(
        f"server: {socket.gethostname()} | gpu: {os.getenv('CUDA_VISIBLE_DEVICES')} | pid: {os.getpid():d}"
    )
    commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    logging.info(f"commit: {commit_hash}")
    logging.info(f"conda env: {os.environ['CONDA_DEFAULT_ENV']}")

    model_path = get_model_path(args.conf, args.ep)
    if not os.path.exists(model_path):
        model_average(args.conf, args.ep)

    logging.info(f"ASR: {model_path}")
    model = ASR(params, phase="test")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # LM
    if lm_weight > 0:
        lm.to(device)
        lm.eval()
    else:
        lm = None

    data_path = get_eval_path(args.data)
    data_tag = (
        args.data
        if args.data_tag == "test" and data_path != args.data
        else args.data_tag
    )
    if data_path is None:
        data_path = params.test_path
    logging.info(f"test data: {data_path}")
    dataset = ASRDataset(params, rel_to_abs_path(data_path), phase="test")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=1,
    )
    vocab = Vocab(rel_to_abs_path(params.vocab_path))

    if args.runtime:
        torch.set_num_threads(1)

        runtimes = []
        for j in range(args.runtime_num_repeats):
            start_time = time.time()
            test(
                model,
                dataloader,
                vocab,
                beam_width,
                len_weight,
                decode_ctc_weight,
                lm,
                lm_weight,
                device,
                eos_id=params.eos_id,
                num_samples=args.runtime_num_samples,
                sample_utt_id=args.utt_id,
            )
            runtime = time.time() - start_time
            runtime /= args.runtime_num_samples
            logging.info(f"Run {(j+1):d} runtime: {runtime:.5f}sec / utt")
            runtimes.append(runtime)

        logging.info(f"Averaged runtime {np.mean(runtimes):.5f}sec on {device.type}")
        return

    if args.utt_id is None:
        results_dir = get_results_dir(args.conf)
        os.makedirs(results_dir, exist_ok=True)
        result_file = f"result_{data_tag}_beam{beam_width}_len{len_weight}_ctc{decode_ctc_weight}_ep{args.ep}.tsv"
        result_path = os.path.join(results_dir, result_file)
        logging.info(f"result: {result_path}")
        if os.path.exists(result_path):
            logging.warning(f"result already exists! (will be overwritten)")

    results = test(
        model,
        dataloader,
        vocab,
        beam_width,
        len_weight,
        decode_ctc_weight,
        lm,
        lm_weight,
        device,
        sample_utt_id=args.utt_id,
    )

    if args.utt_id is None:
        data = pd.DataFrame(results, columns=["utt_id", "token_id", "text", "reftext"])
        data.to_csv(result_path, sep="\t", index=False)

        wer, wer_dict = compute_wers_df(data)
        wer_info = f"WER: {wer:.2f} [D={wer_dict['n_del']:d}, S={wer_dict['n_sub']:d}, I={wer_dict['n_ins']:d}, N={wer_dict['n_ref']:d}]"
        logging.info(wer_info)
        insert_comment(result_path, wer_info)

        return wer, wer_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--utt_id", type=str, default=None)
    parser.add_argument("--runtime", action="store_true")  # measure runtime mode
    parser.add_argument("--runtime_num_samples", type=int, default=20)
    parser.add_argument("--runtime_num_repeats", type=int, default=5)
    #
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--len_weight", type=float, default=None)
    parser.add_argument("--decode_ctc_weight", type=float, default=None)
    parser.add_argument("--lm_weight", type=float, default=0)
    parser.add_argument("--lm_conf", type=str, default=None)
    args = parser.parse_args()

    try:
        main(args)
    except:
        logging.error("***** ERROR occurs in testing *****", exc_info=True)
        logging.error("**********")
