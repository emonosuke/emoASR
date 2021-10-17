import os
import sys

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

import torch.nn as nn
from asr.modeling.encoders.conv import Conv2dEncoder
from asr.modeling.model_utils import make_src_mask
from asr.modeling.transformer import PositionalEncoder, TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, params):
        super(TransformerEncoder, self).__init__()

        self.input_layer = params.input_layer
        self.enc_num_layers = params.enc_num_layers

        input_size = params.feat_dim * params.num_framestacks

        if self.input_layer == "conv2d":
            self.conv = Conv2dEncoder(
                input_dim=input_size, output_dim=params.enc_hidden_size
            )
            input_size = params.enc_hidden_size

        self.pe = PositionalEncoder(input_size, dropout_rate=params.dropout_enc_rate)

        self.transformers = nn.ModuleList()
        for _ in range(params.enc_num_layers):
            self.transformers += [
                TransformerEncoderLayer(
                    params.enc_num_attention_heads,
                    params.enc_hidden_size,
                    params.enc_intermediate_size,
                    params.dropout_enc_rate,
                    params.dropout_attn_rate,
                )
            ]

        # normalize before
        self.norm = nn.LayerNorm(params.enc_hidden_size, eps=1e-12)

    def forward(self, xs, xlens):
        if self.input_layer == "conv2d":
            xs, elens = self.conv(xs, xlens)

        xs = self.pe(xs)
        mask = make_src_mask(elens)
        for layer_id in range(self.enc_num_layers):
            xs, mask = self.transformers[layer_id](xs, mask)
        # normalize before
        xs = self.norm(xs)
        eouts = xs

        return eouts, elens
