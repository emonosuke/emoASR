""" RNN Transducer

Reference:
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/rnn_transducer.py
"""

# TODO: beam search

import logging
import os
import sys

import torch
import torch.nn as nn
import warp_rnnt

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import RNNTAlignDistillLoss, RNNTWordDistillLoss
from asr.modeling.decoders.ctc import CTCDecoder


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
                from asr.modeling.decoders.rnnt_aligner import RNNTForcedAligner

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
            print("CTC is used")
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

    def decode(
        self,
        eouts,
        elens,
        eouts_inter=None,
        beam_width=1,
        len_weight=0,
        decode_ctc_weight=0,
        decode_phone=False,
    ):
        if beam_width <= 1:
            return self._greedy(eouts, elens, decode_ctc_weight)
