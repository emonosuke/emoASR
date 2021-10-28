import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import str2ints

from lm.text_augment import TextAugment

random.seed(0)
eos_id = 2
phone_eos_id = 2


class LMDataset(Dataset):
    def __init__(self, params, data_path, phase="train", size=-1):
        self.data = pd.read_table(data_path, comment="#")[["utt_id", "token_id"]]
        self.lm_type = params.lm_type
        self.add_sos_eos = params.add_sos_eos
        self.phase = phase

        global eos_id
        eos_id = params.eos_id

        if size > 0:
            self.data = self.data[:size]

        if self.lm_type in ["bert", "electra"]:
            self.mask_id = params.mask_id
            # either `num_to_mask` or `mask_proportion` must be specified
            assert hasattr(params, "num_to_mask") ^ hasattr(params, "mask_proportion")
            self.num_to_mask = (
                params.num_to_mask if hasattr(params, "num_to_mask") else -1
            )
            self.mask_proportion = (
                params.mask_proportion if hasattr(params, "mask_proportion") else -1
            )
            self.random_num_to_mask = params.random_num_to_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id = self.data.loc[idx]["utt_id"]
        token_id = str2ints(self.data.loc[idx]["token_id"])
        if self.add_sos_eos:
            token_id = [eos_id] + token_id + [eos_id]

        y = torch.tensor(token_id, dtype=torch.long)

        if self.phase == "train":
            if self.lm_type in ["bert", "electra"]:
                y_in, label = create_masked_lm_label(
                    y,
                    mask_id=self.mask_id,
                    num_to_mask=self.num_to_mask,
                    mask_proportion=self.mask_proportion,
                    random_num_to_mask=self.random_num_to_mask,
                )
            elif self.lm_type == "transformer":
                assert len(y) > 1
                y_in = y[:-1]
                label = y[1:]
        else:
            y_in = y
            label = None

        ylen = y_in.size(0)

        return utt_id, y_in, ylen, label

    @staticmethod
    def collate_fn(batch):
        utt_ids, ys_in, ylens, labels = zip(*batch)

        ret = {}

        ret["utt_ids"] = list(utt_ids)
        ret["ys_in"] = pad_sequence(ys_in, batch_first=True, padding_value=eos_id)
        ret["ylens"] = torch.tensor(ylens)
        if labels[0] is not None:
            ret["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)

        return ret


class P2WDataset(Dataset):
    def __init__(self, params, data_path, phase="train", size=-1):
        self.data = pd.read_table(data_path, comment="#")[
            ["utt_id", "token_id", "phone_token_id"]
        ]
        self.lm_type = params.lm_type
        self.add_sos_eos = params.add_sos_eos
        self.phase = phase

        global eos_id
        eos_id = params.eos_id
        global phone_eos_id
        phone_eos_id = params.phone_eos_id

        if size > 0:
            self.data = self.data[:size]

        if self.phase == "train" and params.text_augment:
            self.textaug = TextAugment(params)
        else:
            self.textaug = None

        # mask y for MLM
        if self.lm_type in ["pelectra"]:
            self.mask_id = params.mask_id
            # either `num_to_mask` or `mask_proportion` must be specified
            assert hasattr(params, "num_to_mask") ^ hasattr(params, "mask_proportion")
            self.num_to_mask = (
                params.num_to_mask if hasattr(params, "num_to_mask") else -1
            )
            self.mask_proportion = (
                params.mask_proportion if hasattr(params, "mask_proportion") else -1
            )
            self.random_num_to_mask = params.random_num_to_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id = self.data.loc[idx]["utt_id"]
        token_id = str2ints(self.data.loc[idx]["token_id"])
        phone_token_id = str2ints(self.data.loc[idx]["phone_token_id"])

        if self.add_sos_eos:
            token_id = [eos_id] + token_id + [eos_id]

        y = torch.tensor(token_id, dtype=torch.long)
        p = torch.tensor(phone_token_id, dtype=torch.long)

        if self.textaug is not None:
            p = self.textaug(p)

        if self.phase == "train":
            if self.lm_type in ["pelectra"]:
                y_in, label = create_masked_lm_label(
                    y,
                    mask_id=self.mask_id,
                    num_to_mask=self.num_to_mask,
                    mask_proportion=self.mask_proportion,
                    random_num_to_mask=self.random_num_to_mask,
                )
        else:
            y_in = y
            label = None

        ylen = y_in.size(0)
        plen = p.size(0)

        return utt_id, p, plen, y_in, ylen, label

    @staticmethod
    def collate_fn(batch):
        utt_ids, ps, plens, ys_in, ylens, labels = zip(*batch)

        ret = {}

        ret["utt_ids"] = list(utt_ids)
        ret["ps"] = pad_sequence(ps, batch_first=True, padding_value=phone_eos_id)
        ret["plens"] = torch.tensor(plens)
        ret["ys_in"] = pad_sequence(ys_in, batch_first=True, padding_value=eos_id)
        ret["ylens"] = torch.tensor(ylens)
        if labels[0] is not None:
            ret["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)

        return ret


def create_masked_lm_label(
    y, mask_id, num_to_mask=-1, mask_proportion=-1, random_num_to_mask=False
):
    y_masked = y.clone()
    masked_lm_label = torch.full(y.shape, dtype=np.int, fill_value=-100)

    cand_indices = [j for j in range(y.size(0)) if y[j] != eos_id]
    random.shuffle(cand_indices)

    if mask_proportion > 0:
        num_to_mask = int(len(cand_indices) * mask_proportion)

    if random_num_to_mask:
        num_to_mask = random.randint(1, num_to_mask)

    mask_indices = sorted(random.sample(cand_indices, num_to_mask))

    for index in mask_indices:
        # everytime, replace with <mask>
        masked_lm_label[index] = y[index]
        y_masked[index] = mask_id

    return y_masked, masked_lm_label


# test `create_masked_lm_label`
if __name__ == "__main__":
    y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    y_masked, masked_lm_label = create_masked_lm_label(y, mask_id=100, mask_prob=0.3)
    print(y_masked)
    print(masked_lm_label)
