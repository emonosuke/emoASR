""" test ASR
"""
import argparse
import logging
import os
import re
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

from lm.modeling.lm import LM
from utils.average_checkpoints import model_average
from utils.configure import load_config
from utils.converters import ints2str, strip_eos
from utils.log import insert_comment
from utils.paths import (get_eval_path, get_model_path, get_results_dir,
                         rel_to_abs_path)
from utils.vocab import Vocab

from asr.datasets import ASRDataset
from asr.metrics import compute_wers_df
from asr.modeling.asr import ASR

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def test_step(
    model, data, beam_width, len_weight, decode_ctc_weight, decode_phone, lm, lm_weight, device
):
    utt_id = data["utt_ids"][0]
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)

    if decode_phone:
        reftext = data["ptexts"][0]
    else:
        reftext = data["texts"][0]

    hyps, scores, _, _ = model.decode(
        xs,
        xlens,
        beam_width,
        len_weight,
        lm=lm,
        lm_weight=lm_weight,
        decode_ctc_weight=decode_ctc_weight,
        decode_phone=decode_phone
    )
    return utt_id, hyps, scores, reftext


def test(
    model,
    dataloader,
    vocab,
    beam_width,
    len_weight,
    decode_ctc_weight,
    decode_phone,
    lm,
    lm_weight,
    device,
    eos_id=2,
    num_samples=-1,
    sample_utt_id=None,
    nbest=False,
):
    rows = []  # utt_id, (score,) token_id, text, reftext

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
            decode_phone,
            lm,
            lm_weight,
            device,
        )
        text = ""

        if nbest:
            for hyp, score in zip(hyps, scores):
                token_id = ints2str(strip_eos(hyp, eos_id))
                text = vocab.ids2text(strip_eos(hyp, eos_id))
                rows.append([utt_id, score, token_id, text, reftext])

            text = vocab.ids2text(strip_eos(hyps[0], eos_id))
        else:
            if len(hyps) < 1:
                token_id = None
                text = ""
                logging.warning(f"cannot decode {utt_id}")
            else:
                # top-1
                token_id = ints2str(strip_eos(hyps[0], eos_id))
                text = vocab.ids2text(strip_eos(hyps[0], eos_id))
            rows.append([utt_id, token_id, text, reftext])

        logging.debug(f"{utt_id}({(i+1):d}/{len(dataloader):d}): {text}")

    return rows


def test_main(args, lm_weight=None, len_weight=None):
    if lm_weight is not None and len_weight is not None:
        logging.info(f"*** lm_weight: {lm_weight:.2f} len_weight {len_weight:.2f}")
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        # make sure all operations are done on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = load_config(args.conf)

    beam_width = args.beam_width if args.beam_width is not None else params.beam_width

    if len_weight is None:
        len_weight = (
            args.len_weight if args.len_weight is not None else params.len_weight
        )
    decode_ctc_weight = (
        args.decode_ctc_weight
        if args.decode_ctc_weight is not None
        else params.decode_ctc_weight
    )
    if lm_weight is None:
        lm_weight = args.lm_weight if args.lm_weight is not None else params.lm_weight

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
        lm_conf = (
            args.lm_conf
            if args.lm_conf is not None
            else rel_to_abs_path(params.lm_conf)
        )
        lm_path = (
            get_model_path(lm_conf, args.lm_ep)
            if args.lm_ep is not None
            else rel_to_abs_path(params.lm_path)
        )

        logging.info(f"LM: {lm_path}")
        lm_params = load_config(lm_conf)
        lm = LM(lm_params, phase="test")
        lm.load_state_dict(torch.load(lm_path, map_location=device))
        lm.to(device)
        lm.eval()
        lm_tag = lm_params.lm_type if args.lm_tag is None else args.lm_tag
    else:
        lm = None
        lm_tag = ""

    data_path = get_eval_path(args.data)
    data_tag = (
        args.data
        if args.data_tag == "test" and data_path != args.data
        else args.data_tag
    )
    if data_path is None:
        data_path = params.test_path
    logging.info(f"test data: {data_path}")
    dataset = ASRDataset(params, rel_to_abs_path(data_path), phase="test", decode_phone=args.decode_phone)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    if args.decode_phone:
        vocab = Vocab(rel_to_abs_path(params.phone_vocab_path), no_subword=True)
    else:
        vocab = Vocab(rel_to_abs_path(params.vocab_path))

    if args.runtime:
        torch.set_num_threads(1)

        runtimes = []
        rtfs = []
        for j in range(args.runtime_num_repeats):
            start_time = time.time()
            results = test(
                model,
                dataloader,
                vocab,
                beam_width,
                len_weight,
                decode_ctc_weight,
                args.decode_phone,
                lm,
                lm_weight,
                device,
                eos_id=params.eos_id,
                num_samples=args.runtime_num_samples,
                sample_utt_id=args.utt_id,
                nbest=args.nbest,
            )
            runtime = time.time() - start_time
            runtime_utt = runtime / args.runtime_num_samples
            utt_ids = [result[0] for result in results]
            wavtime = 0
            for utt_id in utt_ids:
                start_time = int(re.split("_|-", utt_id)[-2]) / args.wavtime_factor
                end_time = int(re.split("_|-", utt_id)[-1]) / args.wavtime_factor
                wavtime += (end_time - start_time)
            rtf = runtime / wavtime
            logging.info(f"Run {(j+1):d} | runtime: {runtime_utt:.5f}sec / utt, wavtime: {wavtime:.5f}sec | RTF: {(rtf):.5f}")
            runtimes.append(runtime_utt)
            rtfs.append(rtf)

        logging.info(f"Averaged runtime {np.mean(runtimes):.5f}sec, RTF {np.mean(rtfs):.5f} on {device.type}")
        return

    if args.utt_id is None:
        results_dir = get_results_dir(args.conf)
        if args.save_dir is not None:
            results_dir = os.path.join(results_dir, args.save_dir)
        os.makedirs(results_dir, exist_ok=True)
        result_file = f"result_{data_tag}_beam{beam_width:d}_len{len_weight:.1f}_ctc{decode_ctc_weight:.1f}_lm{lm_weight:.2f}{lm_tag}_ep{args.ep}.tsv"
        if args.decode_phone:
            result_file = result_file.replace(".tsv", "_phone.tsv")
        if args.nbest:
            result_file = result_file.replace(".tsv", "_nbest.tsv")
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
        args.decode_phone,
        lm,
        lm_weight,
        device,
        sample_utt_id=args.utt_id,
        nbest=args.nbest,
    )

    if args.utt_id is None:
        if args.nbest:
            data = pd.DataFrame(
                results, columns=["utt_id", "score_asr", "token_id", "text", "reftext"]
            )
        else:
            data = pd.DataFrame(
                results, columns=["utt_id", "token_id", "text", "reftext"]
            )
        data.to_csv(result_path, sep="\t", index=False)

        if not args.nbest:
            wer, wer_dict = compute_wers_df(data)
            if args.decode_phone:
                wer_info = f"PER: {wer:.2f} [D={wer_dict['n_del']:d}, S={wer_dict['n_sub']:d}, I={wer_dict['n_ins']:d}, N={wer_dict['n_ref']:d}]"
            else:
                wer_info = f"WER: {wer:.2f} [D={wer_dict['n_del']:d}, S={wer_dict['n_sub']:d}, I={wer_dict['n_ins']:d}, N={wer_dict['n_ref']:d}]"
            logging.info(wer_info)
            insert_comment(result_path, wer_info)

            return lm_weight, len_weight, wer, wer_info

        # TODO: calculate oracle when args.nbest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nbest", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--utt_id", type=str, default=None)
    parser.add_argument("--runtime", action="store_true")  # measure runtime mode
    parser.add_argument("--runtime_num_samples", type=int, default=20)
    parser.add_argument("--runtime_num_repeats", type=int, default=5)
    parser.add_argument("--wavtime_factor", type=float, default=1000)
    #
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--len_weight", type=float, default=None)
    parser.add_argument("--decode_ctc_weight", type=float, default=None)
    parser.add_argument("--lm_weight", type=float, default=None)
    parser.add_argument("--lm_conf", type=str, default=None)
    parser.add_argument("--lm_ep", type=str, default=None)
    parser.add_argument("--lm_tag", type=str, default=None)
    parser.add_argument("--decode_phone", action="store_true")
    args = parser.parse_args()

    try:
        test_main(args)
    except:
        logging.error("***** ERROR occurs in testing *****", exc_info=True)
        logging.error("**********")
