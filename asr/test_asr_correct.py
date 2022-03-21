""" test ASR with LM error correction
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
from lm.modeling.p2w import P2W
from utils.average_checkpoints import model_average
from utils.configure import load_config
from utils.converters import ints2str, np2tensor, tensor2np
from utils.log import insert_comment, print_topk_probs
from utils.paths import (get_eval_path, get_model_path, get_results_dir,
                         rel_to_abs_path)
from utils.vocab import Vocab

from asr.datasets import ASRDataset
from asr.metrics import compute_wers_df
from asr.modeling.asr import ASR

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def aggregate_logits(logits, aligns, blank_id, reduction="max"):
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

        token_probs_allv_tmp.append(tensor2np(torch.softmax(logits[t], dim=-1)))
        token_id_prev = token_id

    token_probs_allv_tmp = np.array(token_probs_allv_tmp)
    if reduction == "max":
        index = np.argmax(token_probs_allv_tmp[:, token_id_prev])
        token_probs_allv.append(token_probs_allv_tmp[index])
        token_probs_v.append(token_probs_allv_tmp[index, token_id_prev])

    return np.array(token_probs_allv), np.array(token_probs_v)


def test_step(
    model,
    lm,
    data,
    blank_id,
    mask_id,
    mask_th,
    phone_mask_th,
    device,
    vocab,
    vocab_size,
    vocab_phone=None,
    debug=False,
    pad_id=0,
    cascade_ctc=False,
    phone_mask_id=None,
):
    utt_id = data["utt_ids"][0]
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    reftext = data["texts"][0]

    # ASR (word)
    hyps, _, logits, aligns = model.decode(xs, xlens, beam_width=0, len_weight=0)
    hyp = np.array(hyps[0])

    if len(hyp) < 1:
        return utt_id, [], [], reftext, 0, 0

    # ASR (phone)
    if vocab_phone is not None:
        hyps_phone, _, logits_phone, aligns_phone = model.decode(
            xs, xlens, beam_width=0, len_weight=0, decode_phone=True
        )
        hyp_phone = np.array(hyps_phone[0])

        if len(hyp_phone) < 1:
            return utt_id, [], [], reftext, 0, 0

    # mask less confident phones
    if phone_mask_th > 0:
        assert phone_mask_id is not None
        phone_probs, phone_probs_v = aggregate_logits(
            logits_phone[0], aligns_phone[0], blank_id
        )
        assert len(hyp_phone) == len(phone_probs)
        assert len(hyp_phone) == len(phone_probs_v)
        phone_mask_indices = phone_probs_v < phone_mask_th
        hyp_phone[phone_mask_indices] = phone_mask_id
    
    if cascade_ctc:
        p = np2tensor(hyp_phone, device=device)
        hyp_cor = lm.decode(ps=p.unsqueeze(0))[0]
        hyp_masked = None
        num_masked = 0
        num_tokens = 0
    else:
        hyp_masked = hyp.copy()
        token_probs, token_probs_v = aggregate_logits(logits[0], aligns[0], blank_id)
        assert len(hyp) == len(token_probs)
        assert len(hyp) == len(token_probs_v)

        # mask less confident tokens
        mask_indices = token_probs_v < mask_th
        hyp_masked[mask_indices] = mask_id

        num_masked = sum(mask_indices)
        num_tokens = len(mask_indices)

        y = np2tensor(hyp_masked, device=device)

        if vocab_phone is None:
            logits = lm(y.unsqueeze(0).to(device))
        else:
            p = np2tensor(hyp_phone, device=device)
            logits = lm(y.unsqueeze(0).to(device), ps=p.unsqueeze(0).to(device))

        lm_token_probs = tensor2np(torch.softmax(logits[0], dim=-1))

        # fusion
        token_probs_mix = (1 - args.lm_weight) * token_probs[
            :, :vocab_size
        ] + args.lm_weight * lm_token_probs[:, :vocab_size]

        y_gen = np.argmax(token_probs_mix, axis=-1)

        hyp_cor = hyp.copy()
        hyp_cor[mask_indices] = y_gen[mask_indices]
        # remove padding
        hyp_cor = [x for x in filter(lambda x: x != pad_id, hyp_cor)]

    if debug:
        print(f"*** {utt_id} ***")
        print(f"Ref.: {reftext}")
        print(f"Hyp.(word): {' '.join(vocab.ids2tokens(hyp))}")
        if vocab_phone is not None:
            print(f"Hyp.(phone): {' '.join(vocab_phone.ids2tokens(hyp_phone))}")
        if not cascade_ctc:
            print(
                f"Hyp.(masked): {' '.join(vocab.ids2tokens(hyp_masked))} ({num_masked:d}/{num_tokens:d} masked)"
            )
            print("ASR probs:")
            token_probs_masked = token_probs[mask_indices]
            print_topk_probs(token_probs_masked, vocab=vocab)
            print("LM probs:")
            lm_token_probs_masked = lm_token_probs[mask_indices]
            print_topk_probs(lm_token_probs_masked, vocab=vocab)
        print(f"Hyp.(correct): {' '.join(vocab.ids2tokens(hyp_cor))}")

    return utt_id, hyp, hyp_cor, reftext, num_masked, num_tokens


def test(
    model,
    lm,
    dataloader,
    vocab,
    vocab_size,
    device,
    blank_id,
    mask_id,
    mask_th,
    phone_mask_th,
    vocab_phone=None,
    debug=False,
    num_samples=-1,
    cascade_ctc=False,
    phone_mask_id=None
):
    rows = []  # utt_id, token_id, text, reftext

    num_masked_all, num_tokens_all = 0, 0

    for i, data in enumerate(dataloader):
        if num_samples > 0 and (i + 1) > num_samples:
            return rows

        utt_id, hyp, hyp_cor, reftext, num_masked, num_tokens = test_step(
            model,
            lm,
            data,
            blank_id,
            mask_id,
            mask_th,
            phone_mask_th,
            device,
            vocab,
            vocab_size,
            vocab_phone=vocab_phone,
            debug=debug,
            cascade_ctc=cascade_ctc,
            phone_mask_id=phone_mask_id
        )
        text = ""
        num_masked_all += num_masked
        num_tokens_all += num_tokens

        if len(hyp) < 1:
            token_id = None
            text = ""
            logging.warning(f"cannot decode {utt_id}")
        else:
            token_id = ints2str(hyp_cor)
            text = vocab.ids2text(hyp_cor)

        rows.append([utt_id, token_id, text, reftext])

        # logging.debug(f"*** {utt_id}({(i+1):d}/{len(dataloader):d}): {text}")

    logging.info(f"masked: {num_masked_all:d} / {num_tokens_all:d}")

    return rows


def test_main(args, lm_weight=None, mask_th=None):
    if lm_weight is not None and mask_th is not None:
        logging.info(f"*** lm_weight: {lm_weight:.2f} mask_th {len_weight:.2f}")
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        # make sure all operations are done on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = load_config(args.conf)

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
    assert lm_params.lm_type in ["bert", "pbert", "pctc"]
    lm_path = get_model_path(args.lm_conf, args.lm_ep)
    logging.info(f"LM: {lm_path}")
    lm_tag = lm_params.lm_type if args.lm_tag is None else args.lm_tag

    if lm_params.lm_type == "bert":
        lm = LM(lm_params, phase="test")
    elif lm_params.lm_type in ["pbert", "pctc"]:
        lm = P2W(lm_params, phase="test")

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
    if lm_params.lm_type in ["pbert", "pctc"]:
        vocab_phone = Vocab(rel_to_abs_path(params.phone_vocab_path))
    else:
        vocab_phone = None

    if args.runtime:
        torch.set_num_threads(1)

        runtimes = []
        rtfs = []
        for j in range(args.runtime_num_repeats):
            start_time = time.time()
            results = test(
                model,
                lm,
                dataloader,
                vocab,
                params.vocab_size,
                device,
                params.blank_id,
                mask_id=lm_params.mask_id,
                mask_th=args.mask_th,
                phone_mask_th=args.phone_mask_th,
                vocab_phone=vocab_phone,
                num_samples=args.runtime_num_samples,
                cascade_ctc=(lm_params.lm_type == "pctc"),
                phone_mask_id=lm_params.phone_mask_id if hasattr(lm_params, "phone_mask_id") else None,
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
            logging.info(f"Run {(j+1):d} | runtime: {runtime_utt:.5f}sec / utt, wavtime {wavtime:.5f}sec | RTF: {(rtf):.5f}")
            runtimes.append(runtime)
            rtfs.append(rtf)

        logging.info(f"Averaged runtime {np.mean(runtimes):.5f}sec, RTF {np.mean(rtfs):.5f} on {device.type}")
        return

    results = test(
        model,
        lm,
        dataloader,
        vocab,
        params.vocab_size,
        device,
        params.blank_id,
        mask_id=lm_params.mask_id,
        mask_th=args.mask_th,
        phone_mask_th=args.phone_mask_th,
        vocab_phone=vocab_phone,
        debug=args.debug,
        cascade_ctc=(lm_params.lm_type == "pctc"),
        phone_mask_id=lm_params.phone_mask_id if hasattr(lm_params, "phone_mask_id") else None
    )

    results_dir = get_results_dir(args.conf)
    os.makedirs(results_dir, exist_ok=True)
    result_file = f"result_{data_tag}_{lm_tag}_corr{args.lm_weight}_maskth{args.mask_th}_pmaskth{args.phone_mask_th}_ep{args.ep}.tsv"
    result_path = os.path.join(results_dir, result_file)
    logging.info(f"result: {result_path}")
    if os.path.exists(result_path):
        logging.warning(f"result already exists! (will be overwritten)")

    data = pd.DataFrame(results, columns=["utt_id", "token_id", "text", "reftext"])
    data.to_csv(result_path, sep="\t", index=False)

    wer, wer_dict = compute_wers_df(data)
    wer_info = f"WER: {wer:.2f} [D={wer_dict['n_del']:d}, S={wer_dict['n_sub']:d}, I={wer_dict['n_ins']:d}, N={wer_dict['n_ref']:d}]"
    logging.info(wer_info)
    insert_comment(result_path, wer_info)

    return wer, wer_info


def main(args):
    test_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("-lm_conf", type=str, required=True)
    parser.add_argument("-lm_ep", type=str, required=True)
    parser.add_argument("--lm_weight", type=float, default=1)
    parser.add_argument("--mask_th", type=float, default=0.9)
    parser.add_argument("--phone_mask_th", type=float, default=-1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--lm_tag", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--runtime", action="store_true")  # measure runtime mode
    parser.add_argument("--runtime_num_samples", type=int, default=20)
    parser.add_argument("--runtime_num_repeats", type=int, default=5)
    parser.add_argument("--wavtime_factor", type=float, default=1000)
    args = parser.parse_args()

    try:
        test_main(args)
    except:
        logging.error("***** ERROR occurs in testing *****", exc_info=True)
        logging.error("**********")
