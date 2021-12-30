import logging
import os
import sys

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from asr.modeling.encoders.conv import Conv2dEncoder


class RNNEncoder(nn.Module):
    def __init__(self, params):
        super(RNNEncoder, self).__init__()

        self.input_layer = params.input_layer
        self.enc_num_layers = params.enc_num_layers

        input_size = params.feat_dim * params.num_framestacks

        if self.input_layer == "conv2d":
            self.conv = Conv2dEncoder(
                input_dim=input_size, output_dim=params.enc_hidden_size
            )
            input_size = params.enc_hidden_size

        self.enc_hidden_sum_fwd_bwd = params.enc_hidden_sum_fwd_bwd

        if self.enc_hidden_sum_fwd_bwd:
            enc_hidden_size = params.enc_hidden_size
        else:
            assert params.enc_hidden_size % 2 == 0
            enc_hidden_size = params.enc_hidden_size // 2
            logging.warning(
                f"enc_hidden_sum_fwd_bwd is False, so LSTM with hidden_size = {enc_hidden_size}"
            )

        self.rnns = nn.ModuleList()
        for _ in range(self.enc_num_layers):
            self.rnns += [
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=params.enc_hidden_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            ]
            input_size = params.enc_hidden_size

        self.dropout = nn.Dropout(p=params.dropout_enc_rate)

    def forward(self, xs, xlens):
        if self.input_layer == "conv2d":
            xs, elens = self.conv(xs, xlens)  # lengths are converted
        elif self.input_layer == "none":
            elens = xlens

        for layer_id in range(self.enc_num_layers):
            self.rnns[layer_id].flatten_parameters()

            xs = pack_padded_sequence(
                xs, elens.cpu(), batch_first=True, enforce_sorted=False
            )
            eouts_pack, _ = self.rnns[layer_id](xs)
            xs, _ = pad_packed_sequence(eouts_pack, batch_first=True)

            if self.enc_hidden_sum_fwd_bwd:
                # NOTE: sum up forward and backward RNN outputs
                # (B, T, enc_hidden_size*2) -> (B, T, enc_hidden_size)
                half = xs.size(-1) // 2
                xs = xs[:, :, :half] + xs[:, :, half:]

            xs = self.dropout(xs)

        eouts = xs
        eouts_inter = None

        return eouts, elens, eouts_inter
