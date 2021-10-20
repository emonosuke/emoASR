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

        ys_in = self.embed(ys_in)
        ys_in = self.pe(ys_in)
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
