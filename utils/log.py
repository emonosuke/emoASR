import os
import sys

import numpy as np
import torch

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from utils.converters import np2tensor


def insert_comment(file_path, comment):
    with open(file_path) as f:
        lines = f.readlines()

    if lines[0] == f"# {comment}\n":
        return

    lines.insert(0, f"# {comment}\n")
    lines.insert(1, "#\n")
    with open(file_path, mode="w") as f:
        f.writelines(lines)


def get_num_parameters(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params, num_params_trainable


def print_topk_probs(probs: np.ndarray, vocab, k=5):
    topk_infos = []
    for prob in probs:
        p_topk, v_topk = torch.topk(np2tensor(prob), k)
        print(
            (
                " | ".join(
                    [
                        f"{vocab.i2t[v.item()]}: {p.item():.3f}"
                        for p, v in zip(p_topk, v_topk)
                    ]
                )
            )
        )
