import logging
import random

import numpy as np

random.seed(0)
np.random.seed(0)


class SpecAugment:
    """ SpecAugment
    """

    def __init__(self, params):
        self.max_mask_freq = params.max_mask_freq
        self.num_masks_freq = params.num_masks_freq

        self.max_mask_time = params.max_mask_time
        self.num_masks_time = params.num_masks_time

        self.replace_with_zero = params.replace_with_zero

        logging.info(f"apply SpecAugment - {vars(self)}")

    def __call__(self, x):
        return self._time_mask(self._freq_mask(x))

    def _freq_mask(self, x):
        """
        Reference:
            https://github.com/espnet/espnet/blob/master/espnet/transform/spec_augment.py
        """
        cloned = x.copy()

        num_mel_channels = cloned.shape[1]
        fs = np.random.randint(0, self.max_mask_freq, size=(self.num_masks_freq, 2))

        for f, mask_end in fs:
            f_zero = random.randrange(0, num_mel_channels - f)
            mask_end += f_zero

            # avoids randrange error if values are equal and range is empty
            if f_zero == f_zero + f:
                continue

            if self.replace_with_zero:
                cloned[:, f_zero:mask_end] = 0
            else:
                cloned[:, f_zero:mask_end] = cloned.mean()
        return cloned

    def _time_mask(self, x):
        """
        Reference:
            https://github.com/espnet/espnet/blob/master/espnet/transform/spec_augment.py
        """
        cloned = x.copy()

        len_spectro = cloned.shape[0]
        ts = np.random.randint(0, self.max_mask_time, size=(self.num_masks_time, 2))

        for t, mask_end in ts:
            # avoid randint range error
            if len_spectro - t <= 0:
                continue
            t_zero = random.randrange(0, len_spectro - t)

            # avoids randrange error if values are equal and range is empty
            if t_zero == t_zero + t:
                continue

            mask_end += t_zero
            if self.replace_with_zero:
                cloned[t_zero:mask_end] = 0
            else:
                cloned[t_zero:mask_end] = cloned.mean()
        return cloned
