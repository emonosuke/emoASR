import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.modeling.model_utils import Swish
from asr.modeling.transformer import MultiHeadedAttention, PositionwiseFeedForward


class RelPositionalEncoder(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, max_len=5000):
        super(RelPositionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.xscale = math.sqrt(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.hidden_size)
        pe_negative = torch.zeros(x.size(1), self.hidden_size)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.hidden_size)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, xs):
        self.extend_pe(xs)
        xs = xs * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - xs.size(1) + 1 : self.pe.size(1) // 2 + xs.size(1),
        ]
        return self.dropout(xs), self.dropout(pos_emb)


class RelMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate):
        super().__init__(
            num_attention_heads, hidden_size, dropout_rate,
        )
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class ConvModule(nn.Module):
    def __init__(self, channels, kernel_size=31):
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )
        self.batch_norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0
        )
        self.glu_act = nn.GLU(dim=1)
        self.swish_act = Swish()

    def forward(self, x):
        # 1. Layernorm is applied before
        x = x.transpose(1, 2)

        # 2. Pointwise Conv
        x = self.pointwise_conv1(x)

        # 3. GLU Conv
        x = self.glu_act(x)

        # 4. 1D Depthwise Conv
        x = self.depthwise_conv(x)

        # 5. Batchnorm
        x = self.batch_norm(x)

        # 6. Swish activation
        x = self.swish_act(x)

        # 7. Pointwise Conv
        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        enc_num_attention_heads,
        enc_hidden_size,
        enc_intermediate_size,
        dropout_enc_rate,
        dropout_attn_rate,
        pos_encode_type="rel",
    ):
        super(ConformerEncoderLayer, self).__init__()

        self.pos_encode_type = pos_encode_type

        if self.pos_encode_type == "abs":
            self.self_attn = MultiHeadedAttention(
                enc_num_attention_heads, enc_hidden_size, dropout_attn_rate
            )
        elif self.pos_encode_type == "rel":
            self.self_attn = RelMultiHeadedAttention(
                enc_num_attention_heads, enc_hidden_size, dropout_attn_rate
            )

        self.conv = ConvModule(enc_hidden_size)
        self.feed_forward = PositionwiseFeedForward(
            enc_hidden_size,
            enc_intermediate_size,
            dropout_enc_rate,
            activation_type="swish",
        )
        self.feed_forward_macaron = PositionwiseFeedForward(
            enc_hidden_size,
            enc_intermediate_size,
            dropout_enc_rate,
            activation_type="swish",
        )

        self.norm_self_attn = nn.LayerNorm(enc_hidden_size)
        self.norm_conv = nn.LayerNorm(enc_hidden_size)
        self.norm_ff = nn.LayerNorm(enc_hidden_size)
        self.norm_ff_macaron = nn.LayerNorm(enc_hidden_size)
        self.norm_final = nn.LayerNorm(enc_hidden_size)

        self.dropout = nn.Dropout(dropout_enc_rate)

    def forward(self, x, mask, pos_emb=None):
        # 1. Feed Forward module
        residual = x
        x = self.norm_ff_macaron(x)
        x = residual + 0.5 * self.dropout(self.feed_forward_macaron(x))

        if self.pos_encode_type == "rel":
            # 2. Multi-Head Self Attention module
            residual = x
            x = self.norm_self_attn(x)
            x_q = x
            x = residual + self.dropout(self.self_attn(x_q, x, x, pos_emb, mask))

            # 3. Convolution module
            residual = x
            x = self.norm_conv(x)
            x = residual + self.dropout(self.conv(x))

        elif self.pos_encode_type == "abs":
            # 2. Convolution module
            residual = x
            x = self.norm_conv(x)
            x = residual + self.dropout(self.conv(x))

            # 3. Multi-Head Self Attention module
            residual = x
            x = self.norm_self_attn(x)
            x_q = x
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        # 4. Feed Forward module
        residual = x
        x = self.norm_ff(x)
        x = residual + 0.5 * self.dropout(self.feed_forward(x))

        # 5. Layernorm
        x = self.norm_final(x)

        return x, mask
