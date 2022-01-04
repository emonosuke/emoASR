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
from asr.modeling.decoders.transformer import CTCDecoder, TransformerDecoder
from asr.modeling.encoders.transformer import TransformerEncoder
from utils.log import get_num_parameters


class P2W(nn.Module):
    def __init__(
        self,
        params,
        phase="train",
        encoder_type=None,
        decoder_type=None,
        return_logits=False,
    ):
        super().__init__()

        self.lm_type = params.lm_type
        logging.info(f"LM type: {self.lm_type}")

        self.encoder = TransformerEncoder(params)

        if decoder_type is None:
            if self.lm_type == "ptransformer":
                self.decoder_type = "transformer"
            elif self.lm_type == "pbert":
                self.decoder_type = "bert"
            elif self.lm_type == "pctc":
                self.decoder_type = "ctc"
        else:
            self.decoder_type = decoder_type

        if self.decoder_type == "transformer":
            self.decoder = TransformerDecoder(params)
        elif self.decoder_type == "bert":
            self.decoder = TransformerDecoder(params, cmlm=True)
        elif self.decoder_type == "ctc":
            self.decoder = CTCDecoder(params)

        self.vocab_size = params.vocab_size
        self.eos_id = params.eos_id
        self.add_sos_eos = params.add_sos_eos

        num_params, num_params_trainable = get_num_parameters(self)
        logging.info(
            f"P2W model #parameters: {num_params} ({num_params_trainable} trainable)"
        )

        self.return_logits = return_logits

    def forward(self, ys=None, ylens=None, labels=None, ps=None, plens=None):
        # DataParallel
        if plens is None:
            plens = torch.tensor([ps.size(1)]).to(ps.device)
        else:
            ps = ps[:, : max(plens)]

        if ylens is None:
            ylens = torch.tensor([ys.size(1)]).to(ys.device)
        else:
            ys = ys[:, : max(ylens)]

        eouts, elens, _ = self.encoder(ps, plens)

        if self.decoder_type == "ctc":
            loss, loss_dict, logits = self.decoder(eouts, elens, ys=ys, ylens=ylens)
            return loss, loss_dict

        # FIXME: take care of `ymask = make_tgt_mask(ylens + 1)`
        if self.decoder_type == "transformer":
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

    def decode(self, ps, plens=None):
        if plens is None:
            plens = torch.tensor([ps.size(1)]).to(ps.device)
        
        eouts, elens, _ = self.encoder(ps, plens)
        hyps, _, _, _ = self.decoder.decode(eouts, elens)
        return hyps
