import os
import random
import sys

import numpy as np
import torch

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import tensor2np


def create_lm_labels(ys_in, ylens_in, ys_out, fill_value=-100):
    assert ys_in.shape == ys_out.shape

    max_ylen = ys_in.size(1)
    lm_labels = []

    for ylen_in, y_out in zip(ylens_in, ys_out):
        lm_label = np.full((max_ylen,), fill_value=fill_value, dtype=np.int)
        lm_label[: ylen_in.item()] = tensor2np(y_out)[: ylen_in.item()]
        lm_labels.append(lm_label)

    return torch.tensor(lm_labels)


def create_masked_lm_labels(
    ys,
    mask_id,
    num_to_mask=-1,
    mask_prob=-1,
    fill_value=-100,
    eos_id=-1,
    pad_id=-1,
    random_num_mask=False,
    mask_indices=None,
    start_mask=0,
    end_mask=10000,
):
    assert (num_to_mask >= 0) ^ (mask_prob >= 0)

    # NOTE: do not mask ys ifself
    ys_masked = ys.clone()

    max_ylen = ys.size(1)
    masked_lm_labels = []

    for i, y in enumerate(ys):
        cand_indices = [
            j for j in range(y.size(0)) if y[j] != eos_id and y[j] != pad_id
        ]
        random.shuffle(cand_indices)

        if mask_prob > 0:
            num_to_mask = max(1, int(len(cand_indices) * mask_prob))

        if random_num_mask:
            num_to_mask_i = random.randint(1, num_to_mask)
        else:
            num_to_mask_i = num_to_mask
        mask_indices = sorted(random.sample(cand_indices, num_to_mask_i))

        masked_lm_label = np.full(max_ylen, dtype=np.int, fill_value=fill_value)

        for index in mask_indices:
            # everytime, replace with <mask>
            masked_lm_label[index] = y[index]
            ys_masked[i, index] = mask_id

        masked_lm_labels.append(masked_lm_label)

    return ys_masked, torch.tensor(masked_lm_labels)
