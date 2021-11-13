import os
import sys

import torch

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

import torch.nn as nn
from asr.modeling.conformer import ConformerEncoderLayer, RelPositionalEncoder
from asr.modeling.encoders.conv import Conv2dEncoder
from asr.modeling.model_utils import make_src_mask
from asr.modeling.transformer import PositionalEncoder, TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    """ Transformer encoder including Conformer
    """

    def __init__(self, params, is_conformer=False):
        super(TransformerEncoder, self).__init__()

        self.input_layer = params.input_layer
        self.enc_num_layers = params.enc_num_layers
        self.pos_encode_type = (
            params.pos_encode_type if hasattr(params, "pos_encode_type") else "abs"
        )
        self.is_conformer = is_conformer

        if self.input_layer == "conv2d":
            input_size = params.feat_dim * params.num_framestacks
            self.conv = Conv2dEncoder(
                input_dim=input_size, output_dim=params.enc_hidden_size
            )
        elif self.input_layer == "embed":
            self.embed = nn.Embedding(params.src_vocab_size, params.enc_hidden_size)
        elif self.input_layer == "linear":
            input_size = params.feat_dim * params.num_framestacks
            self.linear = nn.Linear(input_size, params.enc_hidden_size)
        input_size = params.enc_hidden_size

        if self.pos_encode_type == "abs":
            self.pe = PositionalEncoder(
                input_size, dropout_rate=params.dropout_enc_rate
            )
        elif self.pos_encode_type == "rel":
            assert self.is_conformer
            self.pe = RelPositionalEncoder(
                input_size, dropout_rate=params.dropout_enc_rate
            )

        if self.is_conformer:
            EncoderLayer = ConformerEncoderLayer
        else:
            EncoderLayer = TransformerEncoderLayer

        # TODO: rename to `encoders`
        self.transformers = nn.ModuleList()
        for _ in range(self.enc_num_layers):
            self.transformers += [
                EncoderLayer(
                    enc_num_attention_heads=params.enc_num_attention_heads,
                    enc_hidden_size=params.enc_hidden_size,
                    enc_intermediate_size=params.enc_intermediate_size,
                    dropout_enc_rate=params.dropout_enc_rate,
                    dropout_attn_rate=params.dropout_attn_rate,
                    pos_encode_type=self.pos_encode_type,
                )
            ]

        # normalize before
        # TODO: set `eps` to 1e-5 (default)
        self.norm = nn.LayerNorm(params.enc_hidden_size, eps=1e-12)

        if (
            hasattr(params, "mtl_inter_ctc_weight") and params.mtl_inter_ctc_weight > 0
        ) or (
            hasattr(params, "mtl_phone_ctc_weight") and params.mtl_phone_ctc_weight > 0
        ):
            self.inter_ctc_layer_id = params.inter_ctc_layer_id
        else:
            self.inter_ctc_layer_id = 0

    def forward(self, xs, xlens):
        if self.input_layer == "conv2d":
            xs, elens = self.conv(xs, xlens)
        elif self.input_layer == "embed":
            xs = self.embed(xs)
            elens = xlens
        elif self.input_layer == "linear":
            xs = self.linear(xs)
            elens = xlens

        mask = make_src_mask(elens)
        eouts_inter = None

        if self.pos_encode_type == "abs":
            xs = self.pe(xs)
            pos_emb = None
        elif self.pos_encode_type == "rel":
            xs, pos_emb = self.pe(xs)

        for layer_id in range(self.enc_num_layers):
            xs, mask = self.transformers[layer_id](xs, mask, pos_emb)
            # NOTE: intermediate branches also require normalization.
            if (layer_id + 1) == self.inter_ctc_layer_id:
                eouts_inter = self.norm(xs)

        # normalize before
        xs = self.norm(xs)
        eouts = xs

        return eouts, elens, eouts_inter
