""" Forced alignment with CTC Forward-Backward algorithm

Reference
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
"""

import os
import sys

import torch

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

LOG_0 = -1e10
LOG_1 = 0


def _label_to_path(labels, blank):
    path = labels.new_zeros(labels.size(0), labels.size(1) * 2 + 1).fill_(blank).long()
    path[:, 1::2] = labels
    return path


def _flip_path(path, path_lens):
    """Flips label sequence.
    This function rotates a label sequence and flips it.
    ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[b, t] = path[b, t + path_lens[b]]``
    .. ::
       a b c d .     . a b c d    d c b a .
       e f . . .  -> . . . e f -> f e . . .
       g h i j k     g h i j k    k j i h g
    """
    bs = path.size(0)
    max_path_len = path.size(1)
    rotate = (torch.arange(max_path_len) + path_lens[:, None]) % max_path_len
    return torch.flip(
        path[torch.arange(bs, dtype=torch.int64)[:, None], rotate], dims=[1],
    )


def _flip_label_probability(log_probs, xlens):
    """Flips a label probability matrix.
    This function rotates a label probability matrix and flips it.
    ``log_probs[i, b, l]`` stores log probability of label ``l`` at ``i``-th
    input in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, b, l] = log_probs[i + xlens[b], b, l]``
    """
    xmax, bs, vocab = log_probs.size()
    rotate = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    return torch.flip(
        log_probs[
            rotate[:, :, None],
            torch.arange(bs, dtype=torch.int64)[None, :, None],
            torch.arange(vocab, dtype=torch.int64)[None, None, :],
        ],
        dims=[0],
    )


def _flip_path_probability(cum_log_prob, xlens, path_lens):
    """Flips a path probability matrix.
    This function returns a path probability matrix and flips it.
    ``cum_log_prob[i, b, t]`` stores log probability at ``i``-th input and
    at time ``t`` in a output sequence in ``b``-th batch.
    The rotated matrix ``r`` is defined as
    ``r[i, j, k] = cum_log_prob[i + xlens[j], j, k + path_lens[j]]``
    """
    xmax, bs, max_path_len = cum_log_prob.size()
    rotate_input = (torch.arange(xmax, dtype=torch.int64)[:, None] + xlens) % xmax
    rotate_label = (
        torch.arange(max_path_len, dtype=torch.int64) + path_lens[:, None]
    ) % max_path_len
    return torch.flip(
        cum_log_prob[
            rotate_input[:, :, None],
            torch.arange(bs, dtype=torch.int64)[None, :, None],
            rotate_label,
        ],
        dims=[0, 2],
    )


def _make_pad_mask(seq_lens):
    bs = seq_lens.size(0)
    max_time = seq_lens.max()
    seq_range = torch.arange(0, max_time, dtype=torch.int32, device=seq_lens.device)
    seq_range = seq_range.unsqueeze(0).expand(bs, max_time)
    mask = seq_range < seq_lens.unsqueeze(-1)
    return mask


class CTCForcedAligner(object):
    """
    
    Reference:
        https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
    """

    def __init__(self, blank_id=0):
        self.blank_id = blank_id

    def _computes_transition(
        self, prev_log_prob, path, path_lens, cum_log_prob, y, skip_accum=False
    ):
        """
        prev_log_prob [B, T, vocab]: alpha or beta or gamma
        path [B, T, ]
        path_lens [B]
        cum_log_prob
        y
        """
        bs, max_path_len = path.size()
        mat = prev_log_prob.new_zeros(3, bs, max_path_len).fill_(LOG_0)
        mat[0, :, :] = prev_log_prob
        mat[1, :, 1:] = prev_log_prob[:, :-1]
        mat[2, :, 2:] = prev_log_prob[:, :-2]
        # disable transition between the same symbols
        # (including blank-to-blank)
        same_transition = path[:, :-2] == path[:, 2:]
        mat[2, :, 2:][same_transition] = LOG_0
        log_prob = torch.logsumexp(mat, dim=0)
        outside = torch.arange(max_path_len, dtype=torch.int64,) >= path_lens.unsqueeze(
            1
        )
        log_prob[outside] = LOG_0
        if not skip_accum:
            cum_log_prob += log_prob
        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)

        # print(y[batch_index, path])
        log_prob += y[batch_index, path]
        return log_prob

    def __call__(self, log_probs, elens, ys, ylens):
        """Calculte the best CTC alignment with the forward-backward algorithm.
        """
        bs, xmax, vocab = log_probs.size()
        device = log_probs.device

        # zero padding
        mask = _make_pad_mask(elens.to(device))
        mask = mask.unsqueeze(2).repeat([1, 1, vocab])
        log_probs = log_probs.masked_fill_(mask == 0, 0)
        log_probs = log_probs.transpose(0, 1)  # `[T, B, vocab]`

        path = _label_to_path(ys, self.blank_id)
        path_lens = 2 * ylens.long().cpu() + 1

        ymax = ys.size(1)
        max_path_len = path.size(1)
        assert ys.size() == (bs, ymax), ys.size()
        assert path.size() == (bs, ymax * 2 + 1)

        alpha = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
        alpha[:, 0] = LOG_1
        beta = alpha.clone()
        gamma = alpha.clone()

        batch_index = torch.arange(bs, dtype=torch.int64).unsqueeze(1)
        seq_index = torch.arange(xmax, dtype=torch.int64).unsqueeze(1).unsqueeze(2)
        log_probs_fwd_bwd = log_probs[seq_index, batch_index, path]

        # forward algorithm
        for t in range(xmax):
            alpha = self._computes_transition(
                alpha, path, path_lens, log_probs_fwd_bwd[t], log_probs[t],
            )

        # backward algorithm
        r_path = _flip_path(path, path_lens)
        log_probs_inv = _flip_label_probability(
            log_probs, elens.long().cpu()
        )  # (T, B, vocab)
        log_probs_fwd_bwd = _flip_path_probability(
            log_probs_fwd_bwd, elens.long().cpu(), path_lens
        )  # (T, B, 2*L+1)
        for t in range(xmax):
            beta = self._computes_transition(
                beta, r_path, path_lens, log_probs_fwd_bwd[t], log_probs_inv[t],
            )

        # pick up the best CTC path
        best_aligns = log_probs.new_zeros((bs, xmax), dtype=torch.int64)

        # forward algorithm
        log_probs_fwd_bwd = _flip_path_probability(
            log_probs_fwd_bwd, elens.long().cpu(), path_lens
        )

        for t in range(xmax):
            gamma = self._computes_transition(
                gamma,
                path,
                path_lens,
                log_probs_fwd_bwd[t],
                log_probs[t],
                skip_accum=True,
            )

            # select paths where gamma is valid
            log_probs_fwd_bwd[t] = log_probs_fwd_bwd[t].masked_fill_(
                gamma == LOG_0, LOG_0
            )

            # pick up the best alignment
            offsets = log_probs_fwd_bwd[t].argmax(1)
            for b in range(bs):
                if t <= elens[b] - 1:
                    token_idx = path[b, offsets[b]]
                    best_aligns[b, t] = token_idx

            # remove the rest of paths (select the best path)
            gamma = log_probs.new_zeros(bs, max_path_len).fill_(LOG_0)
            for b in range(bs):
                gamma[b, offsets[b]] = LOG_1

        return best_aligns


if __name__ == "__main__":
    torch.manual_seed(1)
    bs, T, vocab = 2, 8, 3
    logits = torch.rand((bs, T, vocab)) * 10.0
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("probs:", probs)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    elens = torch.tensor([7, 8])
    ys = torch.tensor([[1, 2, 0], [1, 2, 1]])
    ylens = torch.tensor([2, 3])
    aligner = CTCForcedAligner()
    aligns = aligner(log_probs, elens, ys, ylens)
    print(aligns)
