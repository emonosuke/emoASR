""" Forced alignment with RNN-T Forward-Backward algorithm

Reference
    https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/nnet/loss/transducer_loss.py
"""

import math
import time

import torch
from numba import cuda


@cuda.jit(
    "(float32[:,:,:,:], int32[:,:], float32[:,:,:], float32[:], int32[:], int32[:], int32, int32[:,:])"
)
def cu_kernel_forward(log_probs, labels, alpha, log_p, T, U, blank, lock):
    """
    Compute forward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf
    Arguments
    ---------
    log_probs : tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    alpha : tensor
        3D Tensor of (batch x TimeLength x LabelLength) for forward computation.
    log_p : tensor
        1D Tensor of (batch) for forward cost computation.
    T : tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank indice.
    lock : tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """
    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = 0
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the previous computation of Alpha[b,U-1,:]
        # Do the computation over the whole Time sequence on alpha[B,U,:]
        # and then unlock the target U+1 for computation
        while t < T[b]:
            if u == 0:  # init
                if t > 0:
                    alpha[b, t, 0] = alpha[b, t - 1, 0] + log_probs[b, t - 1, 0, blank]
                cuda.atomic.add(lock, (b, u + 1), -1)  # 0 -> -1
                t += 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == 0:
                        alpha[b, 0, u] = (
                            alpha[b, 0, u - 1]
                            + log_probs[b, 0, u - 1, labels[b, u - 1]]
                        )
                    else:
                        # compute emission prob
                        emit = (
                            alpha[b, t, u - 1]
                            + log_probs[b, t, u - 1, labels[b, u - 1]]
                        )
                        # compute no_emission prob
                        no_emit = alpha[b, t - 1, u] + log_probs[b, t - 1, u, blank]
                        # do logsumexp between log_emit and log_no_emit
                        alpha[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u < U[b]:
                        cuda.atomic.add(lock, (b, u + 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)  # -1 -> 0
                    t += 1
        if u == U[b]:
            # for each thread b (utterance)
            # normalize the loss over time
            log_p[b] = (
                alpha[b, T[b] - 1, U[b]] + log_probs[b, T[b] - 1, U[b], blank]
            ) / T[b]


@cuda.jit(
    "(float32[:,:,:,:], int32[:,:], float32[:,:,:], float32[:], int32[:], int32[:], int32, int32[:,:])"
)
def cu_kernel_backward(log_probs, labels, beta, log_p, T, U, blank, lock):
    """
    Compute backward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf
    Arguments
    ---------
    log_probs : tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    beta : tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    log_p : tensor
        1D Tensor of (batch) for backward cost computation.
    T : tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank indice.
    lock : tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """
    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = T[b] - 1
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the next computation of beta[b,U+1,:]
        # Do the computation over the whole Time sequence on beta[B,U,:]
        # and then unlock the target U-1 for computation
        while t >= 0:
            if u == U[b]:  # init
                if t == T[b] - 1:
                    beta[b, t, u] = log_probs[b, t, u, blank]
                else:
                    beta[b, t, u] = beta[b, t + 1, u] + log_probs[b, t, u, blank]
                cuda.atomic.add(lock, (b, u - 1), -1)
                t -= 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == T[b] - 1:
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = (
                            beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        )
                    else:
                        # compute emission prob
                        emit = beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        # compute no_emission prob
                        no_emit = beta[b, t + 1, u] + log_probs[b, t, u, blank]
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u > 0:
                        cuda.atomic.add(lock, (b, u - 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)
                    t -= 1
    if u == 0:
        # for each thread b (utterance)
        # normalize the loss over time
        log_p[b] = beta[b, 0, 0] / T[b]


class RNNTForcedAligner(object):
    def __init__(self, blank_id=0):
        self.blank_id = blank_id

    def __call__(self, log_probs, elens, ys, ylens):
        acts = log_probs.detach()
        labels = ys.int().detach()
        T = elens.int().detach()
        U = ylens.int().detach()

        B, maxT, maxU, _ = acts.shape

        alpha = torch.zeros((B, maxT, maxU), device=acts.device)
        beta = torch.zeros((B, maxT, maxU), device=acts.device)
        lock = torch.zeros((B, maxU), dtype=torch.int32, device=acts.device)
        log_p_alpha = torch.zeros((B,), device=acts.device)
        log_p_beta = torch.zeros((B,), device=acts.device)

        # forward
        cu_kernel_forward[B, maxU](
            acts, labels, alpha, log_p_alpha, T, U, self.blank_id, lock,
        )
        lock = lock * 0
        # backward
        cu_kernel_backward[B, maxU](
            acts, labels, beta, log_p_beta, T, U, self.blank_id, lock
        )
        log_probs_fwd_bwd = alpha + beta

        best_aligns = torch.zeros(
            (B, maxU - 1), dtype=torch.int32, device=log_probs.device
        )

        # alignment
        for b in range(B):
            t, u = 0, 0
            while t + 1 < T[b] and u < U[b]:
                if log_probs_fwd_bwd[b, t + 1, u] > log_probs_fwd_bwd[b, t, u + 1]:
                    t += 1
                else:
                    best_aligns[b, u] = t  # emit y_u
                    u += 1

        return best_aligns


if __name__ == "__main__":
    torch.manual_seed(1)
    log_probs = torch.randn((2, 10, 6, 5)).cuda().log_softmax(dim=-1).requires_grad_()
    labels = torch.Tensor([[1, 2, 1, 2, 0], [1, 2, 1, 2, 3]]).cuda().int()
    T = torch.Tensor([8, 10]).cuda().int()
    U = label_length = torch.Tensor([4, 5]).cuda().int()
    blank = 0

    log_probs = log_probs.detach()
    B, maxT, maxU, A = log_probs.shape
    alpha = torch.zeros((B, maxT, maxU), device=log_probs.device)
    beta = torch.zeros((B, maxT, maxU), device=log_probs.device)
    lock = torch.zeros((B, maxU), dtype=torch.int32, device=log_probs.device)
    log_p_alpha = torch.zeros((B,), device=log_probs.device)
    log_p_beta = torch.zeros((B,), device=log_probs.device)

    cu_kernel_forward[B, maxU](
        log_probs, labels, alpha, log_p_alpha, T, U, blank, lock,
    )
    lock = lock * 0
    cu_kernel_backward[B, maxU](log_probs, labels, beta, log_p_beta, T, U, blank, lock)

    log_probs_fwd_bwd = alpha + beta
