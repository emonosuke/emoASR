import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import str2ints

from lm.text_augment import TextAugment

random.seed(0)
eos_id = 2
phone_eos_id = 2


class LMDataset(Dataset):
    def __init__(self, params, data_path, phase="train", size=-1):
        self.data = pd.read_table(data_path, comment="#")
        
        if params.lm_type in ["electra-disc", "pelectra-disc"]:
            self.data = self.data[["utt_id", "token_id", "error_label"]]
        else:
            if params.bucket_shuffle:
                self.data = self.data[["utt_id", "token_id", "ylen"]]
            else:
                self.data = self.data[["utt_id", "token_id"]]

        len_data = len(self.data)
        self.data = self.data.dropna().reset_index(drop=True)
        if len(self.data) != len_data:
            logging.warning(
                f"nan value in dataset is removed: {len_data:d} -> {len(self.data):d}"
            )

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
        
        if "error_label" in self.data:
            error_label = self.data.loc[idx]["error_label"].split()
            error_label = torch.tensor([e != "C" for e in error_label], dtype=float)
        else:
            error_label = None

        if self.phase == "train":
            if self.lm_type in ["bert", "electra"]:
                y_in, label = create_masked_lm_label(
                    y,
                    mask_id=self.mask_id,
                    num_to_mask=self.num_to_mask,
                    mask_proportion=self.mask_proportion,
                    random_num_to_mask=self.random_num_to_mask,
                )
            elif self.lm_type in ["transformer", "rnn"]:
                assert len(y) > 1
                y_in = y[:-1]
                label = y[1:]
            elif self.lm_type in ["electra-disc", "pelectra-disc"]:
                y_in = y
                label = None
        else:
            y_in = y
            label = None

        ylen = y_in.size(0)

        return utt_id, y_in, ylen, label, error_label

    @staticmethod
    def collate_fn(batch):
        utt_ids, ys_in, ylens, labels, error_labels = zip(*batch)

        ret = {}

        ret["utt_ids"] = list(utt_ids)
        ret["ys_in"] = pad_sequence(ys_in, batch_first=True, padding_value=eos_id)
        ret["ylens"] = torch.tensor(ylens)
        if labels[0] is not None:
            ret["labels"] = pad_sequence(labels, batch_first=True, padding_value=-100)
        if error_labels[0] is not None:
            ret["error_labels"] = pad_sequence(error_labels, batch_first=True, padding_value=-100)

        return ret


class P2WDataset(Dataset):
    def __init__(self, params, data_path, phase="train", size=-1):
        self.data = pd.read_table(data_path, comment="#")

        if params.bucket_shuffle:
            self.data = self.data[
                ["utt_id", "token_id", "phone_token_id", "ylen", "plen"]
            ]
        else:
            self.data = self.data[["utt_id", "token_id", "phone_token_id"]]

        len_data = len(self.data)
        self.data = self.data.dropna().reset_index(drop=True)
        if len(self.data) != len_data:
            logging.warning(
                f"nan value in dataset is removed: {len_data:d} -> {len(self.data):d}"
            )
        
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
        if self.lm_type in ["pelectra", "pbert"]:
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
            self.mask_insert_poisson_lam = (
                params.mask_insert_poisson_lam
                if hasattr(params, "mask_insert_poisson_lam")
                else -1
            )
            self.pad_id = params.pad_id if hasattr(params, "pad_id") else 0

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
            if self.lm_type in ["pelectra", "pbert"]:
                if self.mask_insert_poisson_lam > 0:
                    y_in, label = create_masked_lm_label_insert(
                        y,
                        mask_id=self.mask_id,
                        num_to_mask=self.num_to_mask,
                        mask_proportion=self.mask_proportion,
                        random_num_to_mask=self.random_num_to_mask,
                        insert_poisson_lam=self.mask_insert_poisson_lam,
                        pad_id=self.pad_id,
                    )
                else:
                    y_in, label = create_masked_lm_label(
                        y,
                        mask_id=self.mask_id,
                        num_to_mask=self.num_to_mask,
                        mask_proportion=self.mask_proportion,
                        random_num_to_mask=self.random_num_to_mask,
                    )
            elif self.lm_type == "ptransformer":
                y_in = y[:-1]
                label = y[1:]
            elif self.lm_type == "pctc":
                y_in = y
                label = p
        else:
            y_in = y
            label = None

        plen = p.size(0)
        ylen = y_in.size(0)

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


# TODO common use for LM and P2W (and ASR)
class LMBatchSampler(Sampler):
    def __init__(self, dataset, params, min_batch_size=1):
        if "plen" in dataset.data:
            self.plens = dataset.data["plen"].values
        else:
            self.plens = None
        if "ylen" in dataset.data:
            self.ylens = dataset.data["ylen"].values
        else:
            self.ylens = None

        self.dataset_size = len(self.ylens)

        if hasattr(params, "max_plens_batch"):
            self.max_plens_batch = params.max_plens_batch
        else:
            self.max_plens_batch = 1  # `plens_sum` is always 0

        self.max_ylens_batch = params.max_ylens_batch
        self.batch_size = params.batch_size
        self.min_batch_size = min_batch_size
        self.indices_batches = self._make_batches()

    def _make_batches(self):
        self.index = 0
        indices_batches = []

        while self.index < self.dataset_size:
            indices = []
            plens_sum = 0
            ylens_sum = 0

            while self.index < self.dataset_size:
                plen = self.plens[self.index] if self.plens is not None else 0
                ylen = self.ylens[self.index]

                assert plen <= self.max_plens_batch
                assert ylen <= self.max_ylens_batch
                #print(plens_sum + plen, ylens_sum + ylen)
                if (
                    plens_sum + plen > self.max_plens_batch
                    or ylens_sum + ylen > self.max_ylens_batch
                    or len(indices) + 1 > self.batch_size
                ):
                    break

                indices.append(self.index)
                plens_sum += plen
                ylens_sum += ylen
                self.index += 1

            if len(indices) < self.min_batch_size:
                logging.warning(
                    f"{len(indices)} utterances are skipped because of they are smaller than min_batch_size"
                )
            else:
                indices_batches.append(indices)

        return indices_batches

    def __iter__(self):
        # NOTE: shuffled for each epoch
        random.shuffle(self.indices_batches)
        logging.debug("batches are shuffled in Sampler")

        for indices in self.indices_batches:
            yield indices

    def __len__(self):
        return len(self.indices_batches)


def create_masked_lm_label(
    y, mask_id, num_to_mask=-1, mask_proportion=-1, random_num_to_mask=False,
):
    y_masked = y.clone()
    masked_lm_label = torch.full(y.shape, dtype=np.int, fill_value=-100)

    cand_indices = [j for j in range(y.size(0)) if y[j] != eos_id]
    random.shuffle(cand_indices)

    if mask_proportion > 0:
        num_to_mask = max(int(len(cand_indices) * mask_proportion), 1)

    if random_num_to_mask:
        num_to_mask = random.randint(1, num_to_mask)

    mask_indices = sorted(random.sample(cand_indices, num_to_mask))

    for index in mask_indices:
        # everytime, replace with <mask>
        masked_lm_label[index] = y[index]
        y_masked[index] = mask_id

    return y_masked, masked_lm_label


def create_masked_lm_label_insert(
    y,
    mask_id,
    num_to_mask=-1,
    mask_proportion=-1,
    random_num_to_mask=False,
    insert_poisson_lam=-1,
    pad_id=0,
):
    y_masked, masked_lm_label = create_masked_lm_label(
        y, mask_id, num_to_mask, mask_proportion, random_num_to_mask
    )
    if insert_poisson_lam > 0:
        num_inserts = np.random.poisson(insert_poisson_lam, len(y_masked))
        y_masked_insert = torch.full(
            [len(y_masked) + sum(num_inserts)], dtype=np.int, fill_value=mask_id
        )
        masked_lm_label_insert = torch.full(
            [len(y_masked) + sum(num_inserts)], dtype=np.int, fill_value=pad_id
        )
        index = 0
        for y, label, num_insert in zip(y_masked, masked_lm_label, num_inserts):
            y_masked_insert[index] = y
            masked_lm_label_insert[index] = label
            index = index + 1 + num_insert
    return y_masked_insert, masked_lm_label_insert


# test `create_masked_lm_label`
if __name__ == "__main__":
    # y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # y_masked, masked_lm_label = create_masked_lm_label(y, mask_id=100, mask_proportion=0.3)
    y = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    y_masked, masked_lm_label = create_masked_lm_label_insert(
        y, mask_id=100, mask_proportion=0.3, insert_poisson_lam=0.2, pad_id=0
    )
    print(y_masked)
    print(masked_lm_label)
