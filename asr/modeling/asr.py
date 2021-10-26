""" End-to-End ASR modeling
"""

import logging
import os
import sys

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from asr.modeling.decoders.ctc import CTCDecoder
from asr.modeling.decoders.rnn_transducer import RNNTDecoder
from asr.modeling.decoders.transformer import TransformerDecoder
from asr.modeling.encoders.rnn import RNNEncoder
from asr.modeling.encoders.transformer import TransformerEncoder


class ASR(nn.Module):
    def __init__(self, params, phase="train"):
        super(ASR, self).__init__()

        self.encoder_type = params.encoder_type
        self.decoder_type = params.decoder_type

        logging.info(f"encoder type: {self.encoder_type}")
        if self.encoder_type == "rnn":
            self.encoder = RNNEncoder(params)
        elif self.encoder_type == "transformer":
            self.encoder = TransformerEncoder(params)

        logging.info(f"decoder type: {self.decoder_type}")
        if self.decoder_type == "ctc":
            self.decoder = CTCDecoder(params)
        elif self.decoder_type == "rnn_transducer":
            self.decoder = RNNTDecoder(params, phase)
        elif self.decoder_type == "transformer":
            self.decoder = TransformerDecoder(params)
        # TODO: LAS

        num_params = sum(p.numel() for p in self.parameters())
        num_params_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logging.info(
            f"ASR model #parameters: {num_params} ({num_params_trainable} trainable)"
        )

    def forward(self, xs, xlens, ys, ylens, ys_in, ys_out, soft_labels=None):
        # DataParallel
        xs = xs[:, : max(xlens), :]
        ys = ys[:, : max(ylens)]
        ys_in = ys_in[:, : max(ylens) + 1]
        ys_out = ys_out[:, : max(ylens) + 1]

        eouts, elens = self.encoder(xs, xlens)
        loss, loss_dict = self.decoder(
            eouts, elens, ys, ylens, ys_in, ys_out, soft_labels
        )
        return loss, loss_dict

    def decode(
        self, xs, xlens, beam_width, len_weight, decode_ctc_weight=0,
    ):
        with torch.no_grad():
            eouts, elens = self.encoder(xs, xlens)

        # beam_width == 0 means greedy decoding
        if beam_width == 0:
            assert self.decoder_type in ["ctc", "rnn_transducer"]
            with torch.no_grad():
                hyps, scores, _ = self.decoder.greedy(eouts, elens, decode_ctc_weight)
        else:
            # TODO: support `ctc` and `rnn_transducer`
            assert self.decoder_type in ["transformer"]
            with torch.no_grad():
                hyps, scores = self.decoder.beam_search(
                    eouts, elens, beam_width, len_weight
                )
        return hyps, scores

    def forced_align(self, xs, xlens, decode_ctc_weight=0):
        with torch.no_grad():
            eouts, elens = self.encoder(xs, xlens)
            aligns = self.decoder.forced_align(eouts, elens, decode_ctc_weight)
        return aligns
