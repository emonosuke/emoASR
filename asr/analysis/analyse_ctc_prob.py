""" see frame-level predictions in CTC
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(ROOT_DIR)

from asr.datasets import ASRDataset
from asr.modeling.asr import ASR
from utils.configure import load_config
from utils.paths import get_eval_path, get_model_path, rel_to_abs_path
from utils.vocab import Vocab

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def test(args):
    device = torch.device("cpu")
    torch.set_num_threads(1)
    # make sure all operations are done on cpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    params = load_config(args.conf)
    model_path = get_model_path(args.conf, args.ep)
    
    model = ASR(params, phase="test")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    data_path = get_eval_path(args.data)
    dataset = ASRDataset(params, rel_to_abs_path(data_path), phase="test")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=1,
    )
    vocab = Vocab(rel_to_abs_path(params.vocab_path))

    for i, data in enumerate(dataloader):
        utt_id = data["utt_ids"][0]
        if utt_id != args.utt_id:
            continue

        xs = data["xs"].to(device)
        xlens = data["xlens"].to(device)

        hyps, _, logits, _ = model.decode(xs, xlens)
        probs = torch.softmax(logits, dim=-1)
        print(vocab.ids2text(hyps[0]))
        print("###")

        for i, prob in enumerate(probs[0]):
            print(f"Frame #{i:d}: ", end="")
            p_topk, v_topk = torch.topk(prob, k=args.topk)
            for p, v in zip(p_topk, v_topk):
                print(f"{vocab.id2token(v.item())}({p.item():.3f}) ", end="")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    parser.add_argument("-utt_id", type=str, required=True)
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    test(args)
