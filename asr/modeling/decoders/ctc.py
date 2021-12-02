""" CTC

Reference:
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/ctc.py
"""
import logging
import os
import sys
from itertools import groupby

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import CTCAlignDistillLoss
from asr.modeling.decoders.ctc_aligner import CTCForcedAligner
from utils.converters import ints2str, strip_eos, tensor2np

LOG_0 = -1e10


class CTCDecoder(nn.Module):
    def __init__(self, params):
        super(CTCDecoder, self).__init__()

        self.blank_id = params.blank_id
        self.eos_id = params.eos_id
        self.vocab_size = params.vocab_size

        self.output = nn.Linear(params.enc_hidden_size, self.vocab_size)

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

        if ys is None:
            return logits

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

    def _greedy(self, eouts, elens, decode_phone=False):
        """ Greedy decoding
        """
        bs = eouts.size(0)

        if decode_phone:
            logits = self.phone_output(eouts)
        else:
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

    def _beam_search(
        self, eouts, elens, beam_width=1, len_weight=0, lm=None, lm_weight=0
    ):
        """ Beam search decoding

        Reference:
            https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7
        """
        bs = eouts.size(0)
        assert bs == 1

        logits = self.output(eouts)
        log_probs = torch.log_softmax(logits, dim=-1)

        # init
        beam = {
            "hyp": [self.eos_id],  # <eos> is used for LM
            "score": 0.0,
            "p_b": 0.0,
            "p_nb": LOG_0,
            "score_asr": 0.0,
            "score_lm": 0.0,
            "score_len": 0.0,
        }
        beams = [beam]

        for t in range(elens[0]):
            new_beams = []

            _, v_topk = torch.topk(
                log_probs[:, t],
                k=min(beam_width, self.vocab_size),
                dim=-1,
                largest=True,
                sorted=True,
            )

            if lm_weight > 0:
                # batchify
                hyps_batch = pad_sequence(
                    [torch.tensor(beam["hyp"], device=eouts.device) for beam in beams],
                    batch_first=True,
                )
                hyp_lens_batch = torch.tensor(
                    [len(beam["hyp"]) for beam in beams], device=eouts.device
                )
                lm_log_prob_batch, _ = lm.predict(
                    hyps_batch, hyp_lens_batch, states=None
                )

            for b, beam in enumerate(beams):
                hyp = beam["hyp"]
                p_b = beam["p_b"]  # end with blank
                p_nb = beam["p_nb"]  # end with non-blank
                score_asr = beam["score_asr"]
                score_lm = beam["score_lm"]
                score_len = beam["score_len"]

                # case 1. hyp is not extended (copy the last)
                new_p_b = np.logaddexp(
                    p_b + log_probs[0, t, self.blank_id].item(),
                    p_nb + log_probs[0, t, self.blank_id].item(),
                )
                if len(hyp) > 1:
                    new_p_nb = p_nb + log_probs[0, t, hyp[-1]].item()
                else:
                    new_p_nb = LOG_0
                score_asr = np.logaddexp(new_p_b, new_p_nb)

                new_beams.append(
                    {
                        "hyp": hyp,
                        "score": score_asr + score_lm + score_len,
                        "p_b": new_p_b,
                        "p_nb": new_p_nb,
                        "score_asr": score_asr,
                        "score_lm": score_lm,
                        "score_len": score_len,
                    }
                )

                # case 2. hyp is extended
                new_p_b = LOG_0
                for v in tensor2np(v_topk[0]):
                    p_t = log_probs[0, t, v].item()
                    if v == self.blank_id:
                        continue
                    v_prev = hyp[-1] if len(hyp) > 1 else None
                    if v == v_prev:
                        new_p_nb = p_b + p_t
                    else:
                        new_p_nb = np.logaddexp(p_b + p_t, p_nb + p_t)

                    score_asr = np.logaddexp(new_p_b, new_p_nb)
                    score_len = len_weight * (len(strip_eos(hyp, self.eos_id)) + 1)
                    if lm_weight > 0:
                        score_lm += lm_weight * lm_log_prob_batch[b, v].item()

                    new_beams.append(
                        {
                            "hyp": hyp + [v],
                            "score": score_asr + score_lm + score_len,
                            "p_b": new_p_b,
                            "p_nb": new_p_nb,
                            "score_asr": score_asr,
                            "score_lm": score_lm,
                            "score_len": score_len,
                        }
                    )

                # merge the same hyp
                new_beams = self._merge_ctc_paths(new_beams)

            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[
                :beam_width
            ]

        hyps = [beam["hyp"] for beam in beams]
        scores = [beam["score"] for beam in beams]

        return hyps, scores, logits

    def decode(
        self,
        eouts,
        elens,
        eouts_inter,
        beam_width=1,
        len_weight=0,
        lm=None,
        lm_weight=0,
        decode_ctc_weight=0,
        decode_phone=False,
    ):
        if decode_phone and self.hie_mtl_phone:
            eouts = eouts_inter
        if beam_width <= 1:
            if lm_weight > 0:
                logging.warning("greedy decoding: LM is not used")
            hyps, scores, logits, aligns = self._greedy(eouts, elens, decode_phone)
        else:
            hyps, scores, logits = self._beam_search(
                eouts, elens, beam_width, len_weight, lm, lm_weight
            )
            aligns = None

        return hyps, scores, logits, aligns

    @staticmethod
    def _merge_ctc_paths(beams):
        merged_beams = {}

        for beam in beams:
            hyp = ints2str(beam["hyp"])
            if hyp in merged_beams:
                merged_beams[hyp]["p_b"] = np.logaddexp(
                    merged_beams[hyp]["p_b"], beam["p_b"]
                )
                merged_beams[hyp]["p_nb"] = np.logaddexp(
                    merged_beams[hyp]["p_nb"], beam["p_nb"]
                )
                merged_beams[hyp]["score_asr"] = np.logaddexp(
                    merged_beams[hyp]["score_asr"], beam["score_asr"]
                )
                # NOTE: do not merge `score_lm` and `score_len`
                merged_beams[hyp]["score"] = (
                    merged_beams[hyp]["score_asr"]
                    + merged_beams[hyp]["score_lm"]
                    + merged_beams[hyp]["score_len"]
                )
            else:
                merged_beams[hyp] = beam

        return list(merged_beams.values())
