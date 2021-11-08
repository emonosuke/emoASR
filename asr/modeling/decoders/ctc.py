""" CTC

Reference:
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
"""

# TODO: beam search

import os
import sys
from itertools import groupby

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import CTCAlignDistillLoss
from asr.modeling.decoders.ctc_aligner import CTCForcedAligner


class CTCDecoder(nn.Module):
    def __init__(self, params):
        super(CTCDecoder, self).__init__()

        self.blank_id = params.blank_id

        self.output = nn.Linear(params.enc_hidden_size, params.vocab_size)

        # self.ctc_loss = nn.CTCLoss(blank=self.blank_id, reduction="sum")
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id, reduction="sum", zero_infinity=True
        )

        self.mtl_phone_ctc_weight = (
            params.mtl_phone_ctc_weight
            if hasattr(params, "mtl_phone_ctc_weight")
            else 0
        )
        self.mtl_inter_ctc_weight = (
            params.mtl_inter_ctc_weight
            if hasattr(params, "mtl_inter_ctc_weight")
            else 0
        )
        self.kd_weight = params.kd_weight

        if self.mtl_phone_ctc_weight > 0:
            self.hie_mtl_phone = params.hie_mtl_phone
            self.phone_output = nn.Linear(
                params.enc_hidden_size, params.phone_vocab_size
            )

        if self.kd_weight > 0:
            self.ctc_kd_loss = CTCAlignDistillLoss(
                vocab_size=params.vocab_size,
                blank_id=params.blank_id,
                lsm_prob=params.lsm_prob,
            )
            self.reduce_main_loss_kd = params.reduce_main_loss_kd
            self.forced_aligner = CTCForcedAligner(blank_id=self.blank_id)

    def forward(
        self,
        eouts,
        elens,
        eouts_inter=None,
        ys=None,
        ylens=None,
        ys_in=None,
        ys_out=None,
        soft_labels=None,
        ps=None,
        plens=None,
    ):
        loss = 0
        loss_dict = {}

        logits = self.output(eouts)  # (B, T, vocab_size)

        # NOTE: nn.CTCLoss accepts (T, B, vocab_size) logits
        loss_ctc = self.ctc_loss(
            logits.transpose(1, 0).log_softmax(dim=2), ys, elens, ylens
        ) / logits.size(
            0
        )  # NOTE: nomarlize by B
        loss += loss_ctc  # main loss
        loss_dict["loss_ctc"] = loss_ctc

        if self.mtl_phone_ctc_weight > 0:
            # for elen, plen in zip(elens, plens):
            #     assert elen >= plen

            if self.hie_mtl_phone:
                # https://arxiv.org/abs/1807.06234
                logits_phone = self.phone_output(eouts_inter)  # intermediate layer
            else:
                logits_phone = self.phone_output(eouts)  # final layer

            loss_phone_ctc = self.ctc_loss(
                logits_phone.transpose(1, 0).log_softmax(dim=2), ps, elens, plens
            ) / logits_phone.size(0)

            loss += self.mtl_phone_ctc_weight * loss_phone_ctc

            if self.hie_mtl_phone:
                loss_dict["loss_phone_ctc(inter)"] = loss_phone_ctc
            else:
                loss_dict["loss_phone_ctc"] = loss_phone_ctc

        if self.mtl_inter_ctc_weight > 0:
            logits_inter = self.output(eouts_inter)
            loss_inter_ctc = self.ctc_loss(
                logits_inter.transpose(1, 0).log_softmax(dim=2), ys, elens, ylens
            ) / logits_inter.size(0)

            loss += self.mtl_inter_ctc_weight * loss_inter_ctc
            loss_dict[f"loss_inter_ctc"] = loss_inter_ctc

        if self.kd_weight > 0 and soft_labels is not None:
            log_probs = torch.log_softmax(logits, dim=-1)
            aligns = self.forced_aligner(log_probs, elens, ys, ylens)

            loss_kd = self.ctc_kd_loss(logits, ys, soft_labels, aligns, elens, ylens)
            loss_dict["loss_kd"] = loss_kd

            if self.reduce_main_loss_kd:
                loss = (1 - self.kd_weight) * loss + self.kd_weight * loss_kd
            else:
                loss += self.kd_weight * loss_kd

        loss_dict["loss_total"] = loss

        return loss, loss_dict, logits

    def greedy(self, eouts, elens, decode_ctc_weight=0):
        # NOTE: `decode_ctc_weight` is not used

        bs = eouts.size(0)

        logits = self.output(eouts)
        best_paths = logits.argmax(-1)

        hyps = []
        scores = []
        aligns = []

        for b in range(bs):
            indices = [best_paths[b, t].item() for t in range(elens[b])]
            collapsed_indices = [x[0] for x in groupby(indices)]
            hyp = [x for x in filter(lambda x: x != self.blank_id, collapsed_indices)]
            hyps.append(hyp)
            # TODO
            scores.append(None)
            aligns.append(indices)

        return hyps, scores, logits, aligns
