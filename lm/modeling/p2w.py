""" Phone-to-word modeling
"""

import logging
import os
import sys

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

# same modeling as ASR
from asr.modeling.decoders.transformer import TransformerDecoder
from asr.modeling.encoders.transformer import TransformerEncoder
from utils.log import get_num_parameters


class P2W(nn.Module):
    def __init__(
        self,
        params,
        phase="train",
        encoder_type=None,
        decoder_type=None,
        cmlm=False,
        return_logits=False,
    ):
        super(P2W, self).__init__()

        self.lm_type = params.lm_type
        logging.info(f"LM type: {self.lm_type}")

        self.encoder = TransformerEncoder(params)
        if self.lm_type == "ptransformer":
            self.decoder = TransformerDecoder(params)
        elif self.lm_type == "pbert":
            self.decoder = TransformerDecoder(params, cmlm=True)

        self.vocab_size = params.vocab_size
        self.eos_id = params.eos_id
        self.add_sos_eos = params.add_sos_eos

        num_params, num_params_trainable = get_num_parameters(self)
        logging.info(
            f"P2W model #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        self.return_logits = return_logits

    def forward(self, ys, ylens, labels=None, ps=None, plens=None):
        # DataParallel
        ps = ps[:, : max(plens)]
        ys = ys[:, : max(ylens)]

        eouts, elens, _ = self.encoder(ps, plens)

        # FIXME: take care of `ymask = make_tgt_mask(ylens + 1)`
        if self.lm_type == "ptransformer":
            ylens -= 1

        if labels is None:
            logits = self.decoder(eouts, elens, ys=ys, ylens=ylens, ys_in=ys)
            return logits

        labels = labels[:, : max(ylens)]

        loss, loss_dict, logits = self.decoder(
            eouts, elens, ys=ys, ylens=ylens, ys_in=ys, ys_out=labels
        )

        if self.return_logits:
            return loss, loss_dict, logits
        else:
            return loss, loss_dict
