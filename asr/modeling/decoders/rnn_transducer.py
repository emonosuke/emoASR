""" RNN Transducer

Reference:
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/rnn_transducer.py
"""

import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import warp_rnnt

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import RNNTAlignDistillLoss, RNNTWordDistillLoss
from asr.modeling.decoders.ctc import CTCDecoder
from utils.converters import ints2str


class RNNTDecoder(nn.Module):
    def __init__(self, params, phase="train"):
        super(RNNTDecoder, self).__init__()

        self.dec_num_layers = params.dec_num_layers
        self.dec_hidden_size = params.dec_hidden_size
        self.eos_id = params.eos_id
        self.blank_id = params.blank_id
        self.max_seq_len = 256
        self.mtl_ctc_weight = params.mtl_ctc_weight
        self.kd_weight = params.kd_weight

        # Prediction network (decoder)
        # TODO: -> class
        self.embed = nn.Embedding(params.vocab_size, params.embedding_size)
        self.dropout_emb = nn.Dropout(p=params.dropout_emb_rate)
        self.dropout = nn.Dropout(p=params.dropout_dec_rate)

        self.rnns = nn.ModuleList()
        input_size = params.embedding_size
        for _ in range(self.dec_num_layers):
            self.rnns += [
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=params.dec_hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
            ]
            input_size = params.dec_hidden_size

        # Joint network
        # TODO: -> class
        self.w_enc = nn.Linear(params.enc_hidden_size, params.joint_hidden_size)
        self.w_dec = nn.Linear(params.dec_hidden_size, params.joint_hidden_size)
        self.output = nn.Linear(params.joint_hidden_size, params.vocab_size)

        if self.mtl_ctc_weight > 0:
            self.ctc = CTCDecoder(params)

        if phase == "train":
            logging.info(f"warp_rnnt version: {warp_rnnt.__version__}")

        if self.kd_weight > 0 and phase == "train":
            self.kd_type = params.kd_type
            self.reduce_main_loss_kd = params.reduce_main_loss_kd
            if self.kd_type == "word":
                self.transducer_kd_loss = RNNTWordDistillLoss()
            elif self.kd_type == "align":
                self.transducer_kd_loss = RNNTAlignDistillLoss()

                # cuda init only if forced aligner is used
                from asr.modeling.decoders.rnnt_aligner import \
                    RNNTForcedAligner

                self.forced_aligner = RNNTForcedAligner(blank_id=self.blank_id)

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

        # Prediction network
        douts, _ = self.recurrency(ys_in, dstate=None)

        # Joint network
        logits = self.joint(eouts, douts)  # (B, T, L + 1, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        assert log_probs.size(2) == ys.size(1) + 1

        # NOTE: rnnt_loss only accepts ys, elens, ylens with torch.int
        loss_rnnt = warp_rnnt.rnnt_loss(
            log_probs,
            ys.int(),
            elens.int(),
            ylens.int(),
            average_frames=False,
            reduction="mean",
            blank=self.blank_id,
            gather=False,
        )
        loss += loss_rnnt  # main loss
        loss_dict["loss_rnnt"] = loss_rnnt

        if self.mtl_ctc_weight > 0:
            # NOTE: KD is not applied to auxiliary CTC
            loss_ctc, _, _ = self.ctc(
                eouts=eouts, elens=elens, ys=ys, ylens=ylens, soft_labels=None
            )
            loss += self.mtl_ctc_weight * loss_ctc  # auxiliary loss
            loss_dict["loss_ctc"] = loss_ctc

        if self.kd_weight > 0 and soft_labels is not None:
            if self.kd_type == "word":
                loss_kd = self.transducer_kd_loss(logits, soft_labels, elens, ylens)
            elif self.kd_type == "align":
                aligns = self.forced_aligner(log_probs, elens, ys, ylens)
                loss_kd = self.transducer_kd_loss(
                    logits, ys, soft_labels, aligns, elens, ylens
                )

            loss_dict["loss_kd"] = loss_kd

            if self.reduce_main_loss_kd:
                loss = (1 - self.kd_weight) * loss + self.kd_weight * loss_kd
            else:
                loss += self.kd_weight * loss_kd

        loss_dict["loss_total"] = loss

        return loss, loss_dict, logits

    def joint(self, eouts, douts):
        """ Joint network
        """
        eouts = eouts.unsqueeze(2)  # (B, T, 1, enc_hidden_size)
        douts = douts.unsqueeze(1)  # (B, 1, L, dec_hidden_size)

        out = torch.tanh(self.w_enc(eouts) + self.w_dec(douts))
        out = self.output(out)  # (B, T, L, vocab)

        return out

    def recurrency(self, ys_in, dstate):
        """ Prediction network
        """
        ys_emb = self.dropout_emb(self.embed(ys_in))
        bs = ys_emb.size(0)

        if dstate is None:
            dstate = {}
            dstate["hs"] = torch.zeros(
                self.dec_num_layers, bs, self.dec_hidden_size, device=ys_in.device
            )
            dstate["cs"] = torch.zeros(
                self.dec_num_layers, bs, self.dec_hidden_size, device=ys_in.device
            )

        new_hs, new_cs = [], []
        for layer_id in range(self.dec_num_layers):
            self.rnns[layer_id].flatten_parameters()

            ys_emb, (h, c) = self.rnns[layer_id](
                ys_emb,
                hx=(
                    dstate["hs"][layer_id : layer_id + 1],  # (1, B, dec_hidden_size)
                    dstate["cs"][layer_id : layer_id + 1],
                ),
            )
            new_hs.append(h)
            new_cs.append(c)
            ys_emb = self.dropout(ys_emb)

        new_dstate = {}
        new_dstate["hs"] = torch.cat(new_hs, dim=0)
        new_dstate["cs"] = torch.cat(new_cs, dim=0)

        return ys_emb, new_dstate

    def _greedy(self, eouts, elens, decode_ctc_weight=0):
        """ Greedy decoding
        """
        if decode_ctc_weight == 1:
            # greedy
            return self.ctc.decode(eouts, elens, beam_width=1)

        bs = eouts.size(0)

        hyps = []
        scores = []
        logits = None  # TODO
        aligns = []

        for b in range(bs):
            hyp = []
            align = []

            ys = eouts.new_zeros((1, 1), dtype=torch.long).fill_(self.eos_id)  # <sos>
            dout, dstate = self.recurrency(ys, None)

            T = elens[b]

            t = 0
            while t < T:
                out = self.joint(
                    eouts[b : b + 1, t : t + 1], dout
                )  # (B, 1, 1, vocab_size)
                new_ys = out.squeeze(2).argmax(-1)
                token_id = new_ys[0].item()

                align.append(token_id)

                if token_id == self.blank_id:
                    t += 1
                else:
                    hyp.append(token_id)
                    dout, dstate = self.recurrency(new_ys, dstate)
                if len(hyp) > self.max_seq_len:
                    break

            hyps.append(hyp)
            # TODO
            scores.append(None)
            aligns.append(align)

        return hyps, scores, logits, aligns

    def _beam_search(self, eouts, elens, beam_width=1, len_weight=0, lm=None, lm_weight=0):
        """ Beam search decoding

        Reference:
            ALIGNMENT-LENGTH SYNCHRONOUS DECODING FOR RNN TRANSDUCER
            https://ieeexplore.ieee.org/document/9053040
        """
        bs = eouts.size(0)
        assert bs == 1
        NUM_EXPANDS = 3

        # init
        beam = {
            "hyp": [self.eos_id],   # <sos>
            "score": 0.0,
            "score_asr": 0.0,
            "dstate": {"hs": torch.zeros(self.dec_num_layers, bs, self.dec_hidden_size, device=eouts.device),
                       "cs": torch.zeros(self.dec_num_layers, bs, self.dec_hidden_size, device=eouts.device)}
        }
        beams = [beam]
        
        # time synchronous decoding
        for t in range(eouts.size(1)):
            new_beams = []  # A
            beams_v = beams[:]  # C <- B

            for v in range(NUM_EXPANDS):
                new_beams_v = []  # D

                # prediction network
                ys = torch.zeros((len(beams_v), 1), dtype=torch.int64, device=eouts.device)
                for i, beam in enumerate(beams_v):
                    ys[i] = beam["hyp"][-1]
                dstates_prev = {"hs": torch.cat([beam["dstate"]["hs"] for beam in beams_v], dim=1),
                                "cs": torch.cat([beam["dstate"]["cs"] for beam in beams_v], dim=1)}
                douts, dstates = self.recurrency(ys, dstates_prev)

                # for i, beam in enumerate(beams_v):
                #     beams_v[i]["dstate"] = {"hs": dstates["hs"][:, i:i + 1],
                #                             "cs": dstates["cs"][:, i:i + 1]}
                
                # joint network
                logits = self.joint(eouts[:, t:t + 1], douts)
                scores_asr = torch.log_softmax(logits.squeeze(2).squeeze(1), dim=-1)

                # blank expansion
                for i, beam in enumerate(beams_v):
                    blank_score = scores_asr[i, self.blank_id].item()
                    new_beams.append(beam.copy())
                    new_beams[-1]["score"] += blank_score
                    new_beams[-1]["score_asr"] += blank_score
                    # NOTE: do not update `dstate`

                for i, beam in enumerate(beams_v):
                    beams_v[i]["dstate"] = {"hs": dstates["hs"][:, i:i + 1],
                                            "cs": dstates["cs"][:, i:i + 1]}

                # non-blank expansion
                if v < NUM_EXPANDS - 1:
                    for i, beam in enumerate(beams_v):
                        scores_topk, v_topk = torch.topk(scores_asr[i, 1:], k=beam_width, dim=-1, largest=True, sorted=True)
                        v_topk += 1

                        for k in range(beam_width):
                            v_index = v_topk[k].item()
                            new_beams_v.append({"hyp": beam["hyp"] + [v_index],
                                                "score": beam["score"] + scores_topk[k].item(),
                                                "score_asr": beam["score_asr"] + scores_topk[k].item(),
                                                "dout": None,
                                                "dstate": beam["dstate"]})

                # Local pruning at each expansion
                new_beams_v = sorted(new_beams_v, key=lambda x: x["score"], reverse=True)
                new_beams_v = self._merge_rnnt_paths(new_beams_v)
                beams_v = new_beams_v[:beam_width]  # C <- D
            
            # Local pruning at t-th index
            new_beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)
            new_beams = self._merge_rnnt_paths(new_beams)
            beams = new_beams[:beam_width]  # B <- A
        
        hyps = [beam["hyp"] for beam in beams]
        
        return hyps

    def decode(
        self,
        eouts,
        elens,
        eouts_inter=None,
        beam_width=1,
        len_weight=0,
        lm=None,
        lm_weight=0,
        decode_ctc_weight=0,
        decode_phone=False,
    ):
        if beam_width <= 1:
           hyps, scores, logits, aligns = self._greedy(eouts, elens, decode_ctc_weight)
        else:
            hyps = self._beam_search(
                eouts, elens, beam_width, len_weight, lm, lm_weight
            )
        scores, logits, aligns = None, None, None
        return hyps, scores, logits, aligns

    @staticmethod
    def _merge_rnnt_paths(beams):
        merged_beams = {}

        for beam in beams:
            hyp = ints2str(beam["hyp"])
            if hyp in merged_beams.keys():
                merged_beams[hyp]["score"] = np.logaddexp(merged_beams[hyp]["score"], beam["score"])
            else:
                merged_beams[hyp] = beam

        return list(merged_beams.values())
