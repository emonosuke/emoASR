import logging
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.converters import get_utt_id_nosp, str2ints

from asr.spec_augment import SpecAugment

random.seed(0)
eos_id = 2
phone_eos_id = 2


class ASRDataset(Dataset):
    def __init__(self, params, data_path, phase="train", size=-1, decode_phone=False):
        self.feat_dim = params.feat_dim
        self.num_framestacks = params.num_framestacks
        self.vocab_size = params.vocab_size
        self.lsm_prob = params.lsm_prob

        global eos_id
        eos_id = params.eos_id

        self.phase = phase  # `train` or `test` or `valid`

        if self.phase == "train" and params.spec_augment:
            self.specaug = SpecAugment(params)
        else:
            self.specaug = None

        self.data = pd.read_table(data_path)

        self.mtl_phone_ctc_weight = (
            params.mtl_phone_ctc_weight
            if hasattr(params, "mtl_phone_ctc_weight")
            else 0
        )

        if (self.phase == "train" and self.mtl_phone_ctc_weight > 0) or decode_phone:
            self.data = self.data[
                [
                    "feat_path",
                    "utt_id",
                    "token_id",
                    "text",
                    "xlen",
                    "ylen",
                    "phone_token_id",
                    "phone_text",
                ]
            ]
            global phone_eos_id
            phone_eos_id = params.phone_eos_id
        else:
            self.data = self.data[
                ["feat_path", "utt_id", "token_id", "text", "xlen", "ylen"]
            ]
        
        self.use_kd = params.kd_weight > 0 or (hasattr(params, "inter_kd_weight") and params.inter_kd_weight > 0)

        if self.phase == "train" and self.use_kd:
            with open(params.kd_label_path, "rb") as f:
                self.data_kd = pickle.load(f)
            logging.info(f"kd labels: {params.kd_label_path}")

            self.add_eos = params.decoder_type in ["transformer", "las"]
        else:
            self.data_kd = None

        if size > 0:
            self.data = self.data[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utt_id = self.data.loc[idx]["utt_id"]
        text = self.data.loc[idx]["text"]

        feat_path = self.data.loc[idx]["feat_path"]
        x = np.load(feat_path)[:, : self.feat_dim]

        if self.specaug is not None:
            x = self.specaug(x)

        x = torch.tensor(x, dtype=torch.float)  # float32

        if self.num_framestacks > 1:
            x = self._stack_frames(x, self.num_framestacks)

        xlen = x.size(0)  # `xlen` is based on length after frame stacking

        token_id = str2ints(self.data.loc[idx]["token_id"])
        y = torch.tensor(token_id, dtype=torch.long)  # int64
        ylen = y.size(0)

        if "phone_token_id" in self.data:
            phone_token_id = str2ints(self.data.loc[idx]["phone_token_id"])
            phone_text = self.data.loc[idx]["phone_text"]
            p = torch.tensor(phone_token_id, dtype=torch.long)
            plen = p.size(0)
            ptext = phone_text
        else:
            p, plen, ptext = None, None, None

        # for knowledge distillation
        if self.data_kd is not None:
            utt_id_nosp = get_utt_id_nosp(utt_id)

            if utt_id_nosp in self.data_kd:
                data_kd_utt = self.data_kd[utt_id_nosp]
            else:
                data_kd_utt = []
                logging.warning(f"soft label: {utt_id_nosp} not found")

            soft_label = create_soft_label(
                data_kd_utt, ylen, self.vocab_size, self.lsm_prob, add_eos=self.add_eos
            )
        else:
            soft_label = None

        return utt_id, x, xlen, y, ylen, text, p, plen, ptext, soft_label

    @staticmethod
    def _stack_frames(x, num_framestacks):
        new_len = x.size(0) // num_framestacks
        feat_dim = x.size(1)
        x_stacked = x[0 : new_len * num_framestacks].reshape(
            new_len, feat_dim * num_framestacks
        )

        return x_stacked

    @staticmethod
    def collate_fn(batch):
        utt_ids, xs, xlens, ys_list, ylens, texts, ps, plens, ptexts, soft_labels = zip(*batch)

        ret = {}

        ret["utt_ids"] = list(utt_ids)
        ret["texts"] = list(texts)

        # ys = [[y_1, ..., y_n], ...]
        ret["ys"] = pad_sequence(
            ys_list, batch_first=True, padding_value=eos_id
        )  # NOTE: without <eos>

        # add <sos> and <eos> here
        ys_eos_list = [[eos_id] + y.tolist() + [eos_id] for y in ys_list]

        # ys_in = [[<eos>, y_1, ..., y_n], ...], ys_out = [[y_1, ..., y_n, <eos>], ...]
        ys_in = [torch.tensor(y[:-1], dtype=torch.long) for y in ys_eos_list]
        ys_out = [torch.tensor(y[1:], dtype=torch.long) for y in ys_eos_list]

        ret["xs"] = pad_sequence(xs, batch_first=True)
        ret["xlens"] = torch.tensor(xlens)
        ret["ys_in"] = pad_sequence(
            ys_in, batch_first=True, padding_value=eos_id
        )  # NOTE: <sos> is added
        ret["ys_out"] = pad_sequence(
            ys_out, batch_first=True, padding_value=eos_id
        )  # NOTE: <eos> is added

        # NOTE: ys_in and ys_out have length ylens+1
        ret["ylens"] = torch.tensor(ylens, dtype=torch.long)

        if ps[0] is not None:
            ret["ps"] = pad_sequence(ps, batch_first=True, padding_value=phone_eos_id)
            ret["plens"] = torch.tensor(plens)
            ret["ptexts"] = list(ptexts)

        if soft_labels[0] is not None:
            ret["soft_labels"] = pad_sequence(soft_labels, batch_first=True)

        return ret


class ASRBatchSampler(Sampler):
    def __init__(self, dataset, params, min_batch_size=1):
        self.xlens = dataset.data["xlen"].values
        self.ylens = dataset.data["ylen"].values
        self.dataset_size = len(self.xlens)
        self.max_xlens_batch = params.max_xlens_batch
        self.max_ylens_batch = params.max_ylens_batch
        self.batch_size = params.batch_size
        self.min_batch_size = min_batch_size
        self.indices_batches = self._make_batches()

    def _make_batches(self):
        self.index = 0
        indices_batches = []

        while self.index < self.dataset_size:
            indices = []
            xlens_sum = 0
            ylens_sum = 0

            while self.index < self.dataset_size:
                xlen = self.xlens[self.index]
                ylen = self.ylens[self.index]

                assert xlen <= self.max_xlens_batch
                assert ylen <= self.max_ylens_batch
                if (
                    xlens_sum + xlen > self.max_xlens_batch
                    or ylens_sum + ylen > self.max_ylens_batch
                    or len(indices) + 1 > self.batch_size
                ):
                    break

                indices.append(self.index)
                xlens_sum += xlen
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


def create_soft_label(data_kd_utt, ylen, vocab_size, lsm_prob, add_eos=False):
    if add_eos:
        soft_label = torch.zeros(ylen + 1, vocab_size)  # same length as `ys_out`
    else:
        soft_label = torch.zeros(ylen, vocab_size)

    for i, topk_probs in enumerate(data_kd_utt):
        soft_label[i, :] = lsm_prob / (vocab_size - len(topk_probs))
        for v, prob in topk_probs:
            soft_label[i, v] = prob.astype(np.float64) * (1 - lsm_prob)

    if add_eos:
        soft_label[-1, :] = lsm_prob / (vocab_size - 1)
        soft_label[-1, eos_id] = 1.0 * (1 - lsm_prob)

    return soft_label
