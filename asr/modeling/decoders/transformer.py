""" Transformer
"""

import os
import sys

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import LabelSmoothingLoss
from asr.modeling.decoders.ctc import CTCDecoder
from asr.modeling.model_utils import make_src_mask, make_tgt_mask
from asr.modeling.transformer import PositionalEncoder, TransformerDecoderLayer
from utils.converters import strip_eos


class TransformerDecoder(nn.Module):
    def __init__(self, params):
        super(TransformerDecoder, self).__init__()

        self.embed = nn.Embedding(params.vocab_size, params.dec_hidden_size)
        self.pe = PositionalEncoder(params.dec_hidden_size, params.dropout_dec_rate)
        self.dec_num_layers = params.dec_num_layers

        self.transformers = nn.ModuleList()
        for _ in range(self.dec_num_layers):
            self.transformers += [
                TransformerDecoderLayer(
                    dec_num_attention_heads=params.dec_num_attention_heads,
                    dec_hidden_size=params.dec_hidden_size,
                    dec_intermediate_size=params.dec_intermediate_size,
                    dropout_dec_rate=params.dropout_dec_rate,
                    dropout_attn_rate=params.dropout_attn_rate,
                )
            ]

        self.mtl_ctc_weight = params.mtl_ctc_weight
        if self.mtl_ctc_weight > 0:
            self.ctc = CTCDecoder(params)

        # normalize before
        self.norm = nn.LayerNorm(params.dec_hidden_size, eps=1e-12)
        self.output = nn.Linear(params.dec_hidden_size, params.vocab_size)

        self.lsm_loss = LabelSmoothingLoss(
            vocab_size=params.vocab_size, lsm_prob=params.lsm_prob
        )

        self.kd_weight = params.kd_weight
        # TODO: distillation
        if self.kd_weight > 0:
            pass

        self.eos_id = params.eos_id
        self.max_decode_ylen = params.max_decode_ylen

    def forward(
        self,
        eouts,
        elens,
        ys=None,
        ylens=None,
        ys_in=None,
        ys_out=None,
        soft_labels=None,
    ):
        loss = 0
        loss_dict = {}

        # embedding + positional encoding
        ys_in = self.pe(self.embed(ys_in))
        emask = make_src_mask(elens)
        ymask = make_tgt_mask(ylens + 1)
        for layer_id in range(self.dec_num_layers):
            ys_in, ymask, eouts, emask = self.transformers[layer_id](
                ys_in, ymask, eouts, emask
            )
        ys_in = self.norm(ys_in)  # normalize before
        logits = self.output(ys_in)

        loss_transformer = self.lsm_loss(logits, ys_out, ylens + 1)
        loss += loss_transformer
        loss_dict["loss_transformer"] = loss_transformer

        if self.mtl_ctc_weight > 0:
            # NOTE: KD is not applied to auxiliary CTC
            loss_ctc, _ = self.ctc(eouts, elens, ys, ylens, soft_labels=None)
            loss += self.mtl_ctc_weight * loss_ctc  # auxiliary loss
            loss_dict["loss_ctc"] = loss_ctc

        loss_dict["loss_total"] = loss

        return loss, loss_dict

    def forward_one_step(self, ys_in, ylens_in, eouts):
        ys_in = self.pe(self.embed(ys_in))
        ymask = make_tgt_mask(ylens_in)

        for layer_id in range(self.dec_num_layers):
            ys_in, ymask, eouts, emask = self.transformers[layer_id](
                ys_in, ymask, eouts, None
            )

        ys_in = self.norm(ys_in[:, -1])  # normalize before
        logits = self.output(ys_in)
        return logits

    def beam_search(
        self, eouts, elens, beam_width=1, len_weight=0, decode_ctc_weight=0
    ):
        bs = eouts.size(0)
        assert bs == 1

        # init
        beam = {"hyp": [self.eos_id], "score": 0.0}
        beams = [beam]

        results = []

        for i in range(self.max_decode_ylen):
            new_beams = []

            for beam in beams:
                ys_in = torch.tensor([beam["hyp"]]).to(eouts.device)
                ylens_in = torch.tensor([i + 1]).to(eouts.device)

                scores = torch.log_softmax(
                    self.forward_one_step(ys_in, ylens_in, eouts), dim=-1
                )  # (1, vocab)

                scores_topk, ids_topk = torch.topk(scores, beam_width, dim=1)

                for j in range(beam_width):
                    new_beam = {}
                    new_beam["score"] = beam["score"] + float(scores_topk[0, j])
                    new_beam["hyp"] = beam["hyp"] + [int(ids_topk[0, j])]
                    new_beams.append(new_beam)

            # update `beams`
            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[
                :beam_width
            ]

            beams_extend = []
            for beam in beams:
                # ended beams
                if beam["hyp"][-1] == self.eos_id:
                    # only <eos> is not acceptable
                    if len(strip_eos(beam["hyp"], self.eos_id)) < 1:
                        continue

                    # add length penalty
                    score = beam["score"] + len_weight * len(
                        strip_eos(beam["hyp"], self.eos_id)
                    )

                    results.append({"hyp": beam["hyp"], "score": score})

                    if len(results) >= beam_width:
                        break
                else:
                    beams_extend.append(beam)

            if len(results) >= beam_width:
                break

            beams = beams_extend

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        hyps = [result["hyp"] for result in results]
        scores = [result["score"] for result in results]
        return hyps, scores
