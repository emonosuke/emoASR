""" test ASR
"""
import argparse
import logging
import os
import socket
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import ints2str
from utils.io_utils import load_config
from utils.logger import insert_comment
from utils.path_utils import (
    get_eval_path,
    get_model_path2,
    get_results_dir2,
    rel_to_abs_path,
)
from utils.vocab import Vocab

from asr.dataset2 import ASRDataset
from asr.evaluator.eval_wer import compute_wers2
from asr.models2.asr import ASR

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def test_step(model, data, beam_width, len_weight, decode_ctc_weight, device):
    utt_id = data["utt_ids"][0]
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    reftext = data["texts"][0]

    hyps, scores = model.decode(xs, xlens, beam_width, len_weight, decode_ctc_weight)
    return utt_id, hyps, scores, reftext


def test(model, dataloader, vocab, beam_width, len_p, decode_ctc_weight, device):
    rows = []  # utt_id, token_id, text, reftext

    for data in dataloader:
        utt_id, hyps, scores, reftext = test_step(
            model, data, beam_width, len_p, decode_ctc_weight, device
        )
        text = ""

        if len(hyps) > 0:
            token_id = ints2str(hyps[0])
            text = vocab.ids2text(hyps[0])
            rows.append([utt_id, token_id, text, reftext])

        logging.debug(f"{utt_id}: {text}")

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

    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )
    logging.info(f"***** {' '.join(sys.argv)}")
    logging.info(
        f"server: {socket.gethostname()} | gpu: {os.getenv('CUDA_VISIBLE_DEVICES')} | pid: {os.getpid():d}"
    )
    logging.info(f"torch: {torch.__version__}")

    model_path = get_model_path2(args.conf, args.ep)
    logging.info(f"model: {model_path}")
    model = ASR(params, phase="test")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    data_path = get_eval_path(args.data)
    data_tag = (
        args.data
        if args.data_tag == "test" and data_path != args.data
        else args.data_tag
    )
    if data_path is None:
        data_path = params.test_path
    logging.info(f"data: {data_path}")
    dataset = ASRDataset(params, rel_to_abs_path(data_path), phase="test")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=1,
    )
    vocab = Vocab(rel_to_abs_path(params.vocab_path))

    results_dir = get_results_dir2(args.conf)
    os.makedirs(results_dir, exist_ok=True)
    result_file = f"result_{data_tag}_beam{beam_width}_len{len_weight}_ctc{decode_ctc_weight}_ep{args.ep}.tsv"
    result_path = os.path.join(results_dir, result_file)
    logging.info(f"result: {result_path}")

    results = test(
        model, dataloader, vocab, beam_width, len_weight, decode_ctc_weight, device,
    )

    data = pd.DataFrame(results, columns=["utt_id", "token_id", "text", "reftext"])
    data.to_csv(result_path, sep="\t", index=False)

    wer_all, wer_detail = compute_wers2(data)
    logging.info(wer_detail)
    insert_comment(result_path, wer_detail)

    return wer_all, wer_detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--data_tag", type=str, default="test")
    parser.add_argument("--debug", action="store_true")
    #
    parser.add_argument("--beam_width", type=int, default=None)
    parser.add_argument("--len_weight", type=float, default=None)
    parser.add_argument("--decode_ctc_weight", type=float, default=None)
    args = parser.parse_args()

    try:
        main(args)
    except:
        logging.error("***** ERROR occurs in testing *****", exc_info=True)
        logging.error("**********")
