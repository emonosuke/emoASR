""" CTC score for joint CTC decoding

Reference:
    https://github.com/espnet/espnet/blob/master/espnet/nets/ctc_prefix_score.py
"""

import numpy as np
import six

LOG_0 = -1e10


class CTCPrefixScorer:
    def __init__(self, x, blank_id, eos_id):
        self.blank_id = blank_id
        self.eos_id = eos_id
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state

        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = np.full((self.input_length, 2), LOG_0, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank_id]
        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank_id]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels

        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = np.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = LOG_0
        else:
            r[output_length - 1] = LOG_0

        # prepare forward probabilities for the last label
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = np.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilites log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank_id]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = np.where(cs == self.eos_id)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = np.where(cs == self.blank_id)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = LOG_0

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, np.rollaxis(r, 2)
