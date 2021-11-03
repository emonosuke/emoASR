""" test ASR with LM error correction
"""
import argparse
import logging
import os
import socket
import sys

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
from utils.converters import ints2str, np2tensor, tensor2np
from utils.log import insert_comment
from utils.paths import get_eval_path, get_model_path, get_results_dir, rel_to_abs_path
from utils.vocab import Vocab

from asr.datasets import ASRDataset
from asr.metrics import compute_wers_df
from asr.modeling.asr import ASR

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def aggregate_logits(logits, aligns, blank_id, mask_id, reduction="max"):
    assert logits.size(0) == len(aligns)
    xlen = logits.size(0)

    token_probs_allv = []
    token_probs_allv_tmp = []
    token_probs_v = []

    token_id_prev = None

    for t in range(xlen):
        token_id = aligns[t]

        if token_id == blank_id:
            continue

        if token_id != aligns[t - 1] and token_id_prev is not None:
            token_probs_allv_tmp = np.array(token_probs_allv_tmp)
            if reduction == "max":
                index = np.argmax(token_probs_allv_tmp[:, token_id_prev])
                token_probs_allv.append(token_probs_allv_tmp[index])
                token_probs_v.append(token_probs_allv_tmp[index, token_id_prev])
            token_probs_allv_tmp = []

        probs = torch.softmax(logits[t], dim=-1)
        token_probs_allv_tmp.append(tensor2np(probs))
        token_id_prev = token_id

    token_probs_allv_tmp = np.array(token_probs_allv_tmp)
    if reduction == "max":
        index = np.argmax(token_probs_allv_tmp[:, token_id_prev])
        token_probs_allv.append(token_probs_allv_tmp[index])
        token_probs_v.append(token_probs_allv_tmp[index, token_id_prev])

    return np.array(token_probs_allv), np.array(token_probs_v)


def test_step(model, lm, data, blank_id, mask_id, mask_th, device, vocab):
    utt_id = data["utt_ids"][0]
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    reftext = data["texts"][0]

    # ASR
    hyps, scores, logits, aligns = model.decode(xs, xlens, beam_width=0, len_weight=0)

    if len(hyps[0]) < 1:
        return utt_id, [], [], reftext

    hyp = np.array(hyps[0])
    hyp_masked = hyp.copy()
    token_probs_allv, token_probs_v = aggregate_logits(
        logits[0], aligns[0], blank_id, mask_id
    )
    assert len(hyp) == len(token_probs_allv)
    assert len(hyp) == len(token_probs_v)

    # mask less confident tokens
    mask_indices = token_probs_v < mask_th
    hyp_masked[mask_indices] = mask_id
    # logging.debug(f"{' '.join(vocab.ids2tokens(hyp))}")
    # logging.debug(
    #     f"{' '.join(vocab.ids2tokens(hyp_masked))} ({sum(mask_indices):d}/{len(mask_indices):d} masked)"
    # )

    ys = np2tensor(hyp_masked).unsqueeze(0).to(device)
    logits = lm(ys)
    ys_gen = torch.argmax(logits, dim=-1)

    ys[0, mask_indices] = ys_gen[0, mask_indices]
    hyp_cor = tensor2np(ys[0])
    # logging.debug(f"{' '.join(vocab.ids2tokens(hyp_cor))}")

    return utt_id, hyp, hyp_cor, reftext


def test(model, lm, dataloader, vocab, device, blank_id, mask_id, mask_th):
    rows = []  # utt_id, token_id, text, reftext

    for i, data in enumerate(dataloader):
        utt_id, hyp, hyp_cor, reftext = test_step(
            model, lm, data, blank_id, mask_id, mask_th, device, vocab
        )
        text = ""

        if len(hyp) < 1:
            token_id = None
            text = ""
            logging.warning(f"cannot decode {utt_id}")
        else:
            # token_id = ints2str(hyp)
            # text = vocab.ids2text(hyp)
            token_id = ints2str(hyp_cor)
            text = vocab.ids2text(hyp_cor)

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

    # ASR
    assert params.decoder_type == "ctc"
    model_path = get_model_path(args.conf, args.ep)
    if not os.path.exists(model_path):
        model_average(args.conf, args.ep)
    logging.info(f"ASR: {model_path}")
    model = ASR(params, phase="test")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # LM
    lm_params = load_config(args.lm_conf)
    assert lm_params.lm_type == "bert"
    lm_path = get_model_path(args.lm_conf, args.lm_ep)
    logging.info(f"LM: {lm_path}")
    lm = LM(lm_params, phase="test")
    lm.load_state_dict(torch.load(lm_path, map_location=device))
    lm.to(device)
    lm.eval()

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

    results_dir = get_results_dir(args.conf)
    os.makedirs(results_dir, exist_ok=True)
    result_file = f"result_{data_tag}_cor{lm_params.lm_type}_lm{args.lm_weight}_maskth{args.mask_th}_ep{args.ep}.tsv"
    result_path = os.path.join(results_dir, result_file)
    logging.info(f"result: {result_path}")
    if os.path.exists(result_path):
        logging.warning(f"result already exists! (will be overwritten)")

    results = test(
        model,
        lm,
        dataloader,
        vocab,
        device,
        blank_id=params.blank_id,
        mask_id=lm_params.mask_id,
        mask_th=args.mask_th,
    )

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
    parser.add_argument("-lm_conf", type=str, required=True)
    parser.add_argument("-lm_ep", type=str, required=True)
    parser.add_argument("--lm_weight", type=int, default=1)
    parser.add_argument("--mask_th", type=float, default=0.9)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        main(args)
    except:
        logging.error("***** ERROR occurs in testing *****", exc_info=True)
        logging.error("**********")
