""" Generate soft labels using LM for distillation

{ utt_id: [[(vocab, prob), ... ], [...], ...], ... }
"""

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from lm.modeling.lm import LM
from utils.converters import str2ints, tensor2np
from utils.io_utils import load_config
from utils.paths import get_model_path

BATCH_SIZE = 100
LOG_STEP = 10000
SAVE_STEP = 10000


def make_lm_label(
    df,
    model,
    device,
    save_path,
    topk=8,
    temp=3.0,
    add_sos_eos=False,
    eos_id=2,
    max_seq_len=256,
):
    labels = {}

    utt_ids, ys, ylens, start_poss, end_poss = [], [], [], [], []  # batch

    for i, row in enumerate(df.itertuples()):
        ids = str2ints(row.token_id)

        if add_sos_eos:
            if len(ids) <= max_seq_len - 2:
                ids = [eos_id] + ids + [eos_id]
                start_pos = row.start_pos + 1
                end_pos = row.end_pos + 1
            else:
                # reduce context
                ids = [eos_id] + ids[1:-1] + [eos_id]
                start_pos = row.start_pos
                end_pos = row.end_pos
        else:
            start_pos = row.start_pos
            end_pos = row.end_pos

        y = torch.tensor(ids)
        ylen = len(ids)

        utt_ids.append(row.utt_id)
        ys.append(y)
        ylens.append(ylen)
        start_poss.append(start_pos)
        end_poss.append(end_pos)

        # batchify
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(df):
            bs = len(ys)
            ys_pad = pad_sequence(ys, batch_first=True).to(device)
            ylens = torch.tensor(ylens).to(device)

            with torch.no_grad():
                logits = model(ys_pad, ylens)

            for b in range(bs):
                utt_id = utt_ids[b]
                start_pos = start_poss[b]
                end_pos = end_poss[b]
                y = ys[b]

                for pos in range(start_pos, end_pos):
                    if pos == 0:
                        v_topk = np.array([y[pos]])
                        p_topk = np.array([1.0])
                        logging.warning(f"hard label is used: {v_topk}")
                    else:
                        o_sorted, v_sorted = torch.sort(
                            logits[b, pos - 1], descending=True
                        )
                        o_topk = o_sorted[:topk]
                        v_topk = tensor2np(v_sorted[:topk])
                        p_topk = tensor2np(torch.softmax((o_topk / temp), dim=0))

                    label = []
                    for v, p in zip(v_topk, p_topk):
                        # NOTE: do not add <eos> to soft labels
                        if add_sos_eos and v == eos_id:
                            continue
                        label.append((v, p))

                    if utt_id not in labels:  # first token in utterance
                        labels[utt_id] = [label]
                    else:
                        labels[utt_id].append(label)

            utt_ids, ys, ylens, start_poss, end_poss = [], [], [], [], []

        if (i + 1) % LOG_STEP == 0:
            logging.info(f"{(i+1):>4} / {len(df):>4}")
        if (i + 1) == SAVE_STEP:
            save_tmp_path = save_path + ".tmp"
            with open(save_tmp_path, "wb") as f:
                pickle.dump(labels, f)
            logging.info(f"pickle is saved to {save_tmp_path}")

    with open(save_path, "wb") as f:
        pickle.dump(labels, f)
    logging.info(f"pickle is saved to {save_path}")


def make_bert_label(
    df,
    model,
    device,
    save_path,
    topk=8,
    temp=3.0,
    add_sos_eos=False,
    eos_id=2,
    max_seq_len=256,
):
    labels = {}

    utt_ids, ys, ylens, mask_poss = [], [], [], []  # batch

    for i, row in enumerate(df.itertuples()):
        ids = str2ints(row.token_id)

        if add_sos_eos:
            if len(ids) <= max_seq_len - 2:
                ids = [eos_id] + ids + [eos_id]
                mask_poss.append(row.mask_pos + 1)
            else:
                # reduce context
                ids = [eos_id] + ids[1:-1] + [eos_id]
                mask_poss.append(row.mask_pos)
        else:
            mask_poss.append(row.mask_pos)

        assert len(ids) <= max_seq_len

        y = torch.tensor(ids)
        ylen = len(ids)

        utt_ids.append(row.utt_id)
        ys.append(y)
        ylens.append(ylen)

        # batchify
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == len(df):
            bs = len(ys)
            ys_pad = pad_sequence(ys, batch_first=True).to(device)
            ylens = torch.tensor(ylens).to(device)

            with torch.no_grad():
                logits = model(ys_pad, ylens)

            for b in range(bs):
                utt_id = utt_ids[b]
                mask_pos = mask_poss[b]

                # print("mask_pos:", mask_pos)

                o_sorted, v_sorted = torch.sort(logits[b, mask_pos], descending=True)
                o_topk = o_sorted[:topk]
                v_topk = tensor2np(v_sorted[:topk])

                p_topk = tensor2np(torch.softmax((o_topk / temp), dim=0))

                label = []
                for v, p in zip(v_topk, p_topk):
                    # NOTE: do not add <eos> to soft labels
                    if add_sos_eos and v == eos_id:
                        continue

                    label.append((v, p))

                if utt_id not in labels:  # first token in utterance
                    labels[utt_id] = [label]
                else:
                    labels[utt_id].append(label)

            utt_ids, ys, ylens, mask_poss = [], [], [], []

        if (i + 1) % LOG_STEP == 0:
            logging.info(f"step {(i+1):>4} / {len(df):>4} done")
        if (i + 1) == SAVE_STEP:
            save_tmp_path = save_path + ".tmp"
            with open(save_tmp_path, "wb") as f:
                pickle.dump(labels, f)
            logging.info(f"pickle is saved to {save_tmp_path}")

    with open(save_path, "wb") as f:
        pickle.dump(labels, f)
    logging.info(f"pickle is saved to {save_path}")


def main(args):
    params = load_config(args.lm_conf)

    log_path = args.data_path.replace(
        ".tsv", f"_{params.lm_type}_ep{args.lm_ep}_top{args.topk:d}_temp{args.temp}.log"
    )
    save_path = args.data_path.replace(
        ".tsv", f"_{params.lm_type}_ep{args.lm_ep}_top{args.topk:d}_temp{args.temp}.pkl"
    )
    if args.add_sos_eos:
        save_path = save_path.replace(".pkl", "_addeos.pkl")

    if args.debug:
        logging.basicConfig(
            format="%(asctime)s %(message)s", level=logging.DEBUG
        )  # to stdout
    else:
        logging.basicConfig(
            format="%(asctime)s %(message)s", filename=log_path, level=logging.INFO
        )

    logging.info(f"# {' '.join(sys.argv)}")

    if args.cpu:
        device = torch.device("cpu")
        torch.set_num_threads(1)
        # make sure all operations are done on cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_table(args.data_path)
    logging.info(f"Read tsv ({len(df)} samples)")

    logging.info(f"pickle will be saved to: {save_path}")

    model_path = get_model_path(args.lm_conf, args.lm_ep)
    logging.info(f"model: {model_path}")
    model = LM(params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    if params.lm_type == "bert":
        make_bert_label(
            df,
            model,
            device,
            save_path,
            topk=args.topk,
            temp=args.temp,
            add_sos_eos=args.add_sos_eos,
            eos_id=params.eos_id,
        )
    elif params.lm_type == "transformer":
        make_lm_label(
            df,
            model,
            device,
            save_path,
            topk=args.topk,
            temp=args.temp,
            add_sos_eos=args.add_sos_eos,
            eos_id=params.eos_id,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str)
    parser.add_argument("-lm_conf", type=str, required=True)
    parser.add_argument("-lm_ep", type=str, required=True)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--temp", type=float, default=3.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--add_sos_eos", action="store_true")
    args = parser.parse_args()

    main(args)
