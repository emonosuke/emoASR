import logging
import random

import numpy as np

random.seed(0)
np.random.seed(0)


class SpecAugment:
    """ SpecAugment

    Reference:
        https://arxiv.org/abs/1904.08779
    """

    def __init__(self, params):
        self.max_mask_freq = params.max_mask_freq
        self.num_masks_freq = params.num_masks_freq

        if hasattr(params, "max_mask_time_ratio"):
            # Adaptive SpecAugment
            # https://arxiv.org/pdf/1912.05533.pdf
            self.adaptive_specaug = True
            self.max_mask_time_ratio = params.max_mask_time_ratio
            self.num_masks_time_ratio = params.num_masks_time_ratio
        else:
            self.adaptive_specaug = False
            self.max_mask_time = params.max_mask_time
            self.num_masks_time = params.num_masks_time

        self.replace_with_zero = params.replace_with_zero

        logging.info(f"apply SpecAugment - {vars(self)}")

    def __call__(self, x: np.ndarray):
        return self._time_mask(self._freq_mask(x))

    def _freq_mask(self, x: np.ndarray):
        """
        Reference:
            https://github.com/espnet/espnet/blob/master/espnet/transform/spec_augment.py
        """
        cloned = x.copy()
        fdim = cloned.shape[1]

        fs = np.random.randint(0, self.max_mask_freq, size=(self.num_masks_freq, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, fdim - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            if self.replace_with_zero:
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned

    def _time_mask(self, x: np.ndarray):
        """
        Reference:
            https://github.com/espnet/espnet/blob/master/espnet/transform/spec_augment.py
        """
        cloned = x.copy()
        xlen = cloned.shape[0]

        if self.adaptive_specaug:
            max_mask_time = min(20, round(xlen * self.max_mask_time_ratio))
            num_masks_time = min(20, round(xlen * self.num_masks_time_ratio))
        else:
            max_mask_time = self.max_mask_time
            num_masks_time = self.num_masks_time

        ts = np.random.randint(0, max_mask_time, size=(num_masks_time, 2))

        for t, mask_end in ts:
            # avoid randint range error
            if xlen - t <= 0:
                continue
            t_zero = random.randrange(0, xlen - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if self.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned
