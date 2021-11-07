""" test LM on perplexity (PPL)
"""
import argparse
import logging
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import tensor2np
from utils.io_utils import load_config
from utils.log import insert_comment
from utils.paths import get_eval_path, get_model_path, rel_to_abs_path
from utils.vocab import Vocab

from lm.datasets import LMDataset, P2WDataset
from lm.modeling.lm import LM
from lm.modeling.p2w import P2W

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

LOG_STEP = 100


def ppl_lm(dataloader, model, device, vocab):
    cnt = 0
    sum_logprob = 0

    for i, data in enumerate(dataloader):
        if (i + 1) % LOG_STEP == 0:
            logging.info(
                f"{(i+1):>4} / {len(dataloader):>4} PPL: {math.exp(sum_logprob/cnt):.3f}"
            )
        utt_id = data["utt_ids"][0]
        ys = data["ys_in"]
        ys_in = ys[:, :-1].to(device)
        ys_out = ys[:, 1:].to(device)
        ylens = data["ylens"].to(device) - 1
        assert ys.size(0) == 1

        if ys.size(1) <= 1:
            logging.warning(f"skip {utt_id}")
            continue

        with torch.no_grad():
            logits = model(ys_in, ylens, labels=None)
        logprobs = torch.log_softmax(logits, dim=-1)

        for logprob, label in zip(logprobs[0], ys_out[0]):
            sum_logprob -= logprob[label].item()
            cnt += 1

    ppl = math.exp(sum_logprob / cnt)

    return cnt, ppl


def ppl_masked_lm(dataloader, model, device, mask_id, max_seq_len, vocab):
    cnt = 0
    sum_logprob = 0

    for i, data in enumerate(dataloader):
        if (i + 1) % LOG_STEP == 0:
            logging.info(
                f"{(i+1):>4} / {len(dataloader):>4} PPL: {math.exp(sum_logprob/cnt):.3f}"
            )
        utt_id = data["utt_ids"][0]
        ys = data["ys_in"].to(device)  # not masked
        ylens = data["ylens"].to(device)
        assert ys.size(0) == 1

        # for P2WDataset
        ps = data["ps"].to(device) if "ps" in data else None
        plens = data["plens"].to(device) if "plens" in data else None

        if ys.size(1) > max_seq_len:
            logging.warning(f"input length longer than {max_seq_len:d} skip")
            continue

        if args.print_probs:
            print("********************")
            print(f"{utt_id}: {vocab.ids2text(tensor2np(ys[0]))}")

        for mask_pos in range(ys.size(1)):
            ys_masked = ys.clone()
            label = ys[0, mask_pos]
            ys_masked[0, mask_pos] = mask_id

            with torch.no_grad():
                logits = model(ys_masked, ylens, labels=None, ps=ps, plens=plens)

            if args.print_probs:
                print(vocab.ids2text(tensor2np(ys_masked[0])))
                # TODO: print phones

                probs = torch.softmax(logits, dim=-1)
                p_topk, v_topk = torch.topk(probs[0, mask_pos], k=5)
                print(
                    f"{vocab.i2t[label.item()]} || "
                    + " | ".join(
                        [
                            f"{vocab.i2t[v.item()]}: {p.item():.2f}"
                            for p, v in zip(p_topk, v_topk)
                        ]
                    )
                )

            logprobs = torch.log_softmax(logits, dim=-1)
            sum_logprob -= logprobs[0, mask_pos, label].item()
            cnt += 1

    ppl = math.exp(sum_logprob / cnt)

    return cnt, ppl


def test(model, dataloader, params, device, vocab):
    if params.lm_type in ["bert", "pbert"]:
        cnt, ppl = ppl_masked_lm(
            dataloader,
            model,
            device,
            mask_id=params.mask_id,
            max_seq_len=params.max_seq_len,
            vocab=vocab,
        )
    elif params.lm_type == "transformer":
        cnt, ppl = ppl_lm(dataloader, model, device, vocab=vocab)

    logging.info(f"{cnt} tokens")
    return ppl


def main(args):
    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        # make sure all operations are done on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    params = load_config(args.conf)

    data_path = get_eval_path(args.data)
    if data_path is None:
        data_path = params.test_path
    logging.info(f"test data: {data_path}")

    with open(rel_to_abs_path(data_path)) as f:
        lines = f.readlines()
        logging.info(lines[0])

    if params.lm_type in ["pelectra", "ptransformer", "pbert"]:
        dataset = P2WDataset(params, rel_to_abs_path(data_path), phase="test")
    else:
        dataset = LMDataset(params, rel_to_abs_path(data_path), phase="test")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=1,
    )

    vocab = Vocab(rel_to_abs_path(params.vocab_path))

    model_path = get_model_path(args.conf, args.ep)
    logging.info(f"model: {model_path}")

    if params.lm_type in ["ptransformer", "pbert"]:
        model = P2W(params)
    else:
        model = LM(params)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    ppl = test(model, dataloader, params, device, vocab)

    ppl_info = f"PPL: {ppl:.2f} (conf: {args.conf})"
    logging.info(ppl_info)

    if args.comment:
        insert_comment(args.data, ppl_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--print_probs", action="store_true")
    parser.add_argument("--comment", action="store_true")
    args = parser.parse_args()
    main(args)
