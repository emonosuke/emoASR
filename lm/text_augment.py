import random
from collections import namedtuple

import numpy as np
import torch

random.seed(0)
np.random.seed(0)


class TextAugment:
    """ TextAugment

    Reference:
        - https://arxiv.org/abs/2011.08469
    """

    def __init__(self, params):
        self.max_mask_prob = params.textaug_max_mask_prob
        self.max_replace_prob = params.textaug_max_replace_prob
        self.phone_vocab_size = params.src_vocab_size
        self.eos_id = params.phone_eos_id
        self.mask_id = params.phone_mask_id

    def __call__(self, x):
        return self._text_replace(self._text_mask(x))

    def _text_mask(self, x):
        x_masked = x.clone()
        if self.max_mask_prob <= 0:
            return x_masked

        num_to_mask = random.randint(0, int(len(x) * self.max_mask_prob))
        cand_indices = [j for j in range(len(x)) if x[j] != self.eos_id]
        mask_indices = random.sample(cand_indices, min(len(cand_indices), num_to_mask))
        x_masked[mask_indices] = self.mask_id
        return x_masked

    def _text_replace(self, x):
        x_replaced = x.clone()
        if self.max_replace_prob <= 0:
            return x_replaced

        num_to_replace = random.randint(0, int(len(x) * self.max_replace_prob))
        cand_indices = [j for j in range(len(x)) if x[j] != self.eos_id]
        replace_indices = random.sample(
            cand_indices, min(len(cand_indices), num_to_replace)
        )
        cand_vocab = [j for j in range(self.phone_vocab_size) if j != self.eos_id]
        replaced_ids = random.choices(cand_vocab, k=num_to_replace)
        x_replaced[replace_indices] = torch.tensor(replaced_ids, dtype=torch.long)
        return x_replaced


if __name__ == "__main__":
    p = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2])

    params = namedtuple(
        "Params",
        [
            "textaug_max_mask_prob",
            "textaug_max_replace_prob",
            "src_vocab_size",
            "phone_eos_id",
            "mask_id",
        ],
    )
    params.textaug_max_mask_prob = 0.2
    params.textaug_max_replace_prob = 0.2
    params.src_vocab_size = 11
    params.phone_eos_id = 2
    params.mask_id = 10
    #
    textaug = TextAugment(params)
    p = textaug(p)
    print(p)
