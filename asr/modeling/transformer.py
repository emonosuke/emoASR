import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.modeling.model_utils import Swish


class PositionalEncoder(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.xscale = math.sqrt(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, xs):
        if self.pe is not None:
            if self.pe.size(1) >= xs.size(1):
                if self.pe.dtype != xs.dtype or self.pe.device != xs.device:
                    self.pe = self.pe.to(dtype=xs.dtype, device=xs.device)
                return
        pe = torch.zeros(xs.size(1), self.hidden_size)
        position = torch.arange(0, xs.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=xs.device, dtype=xs.dtype)

    def forward(self, xs):
        self.extend_pe(xs)
        # ASR
        xs = xs * self.xscale + self.pe[:, : xs.size(1)]
        return self.dropout(xs)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, dropout_rate):
        super(MultiHeadedAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_size // num_attention_heads
        self.h = num_attention_heads
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )

            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self, input_size, intermediate_size, dropout_rate, activation_type="relu"
    ):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(input_size, intermediate_size)
        self.w2 = nn.Linear(intermediate_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)

        if activation_type == "relu":
            # TODO: rename to `act`
            self.activation = nn.ReLU()
        elif activation_type == "swish":
            self.activation = Swish()  # for Conformer

    def forward(self, x):
        return self.w2(self.dropout(self.activation(self.w1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        enc_num_attention_heads,
        enc_hidden_size,
        enc_intermediate_size,
        dropout_enc_rate,
        dropout_attn_rate,
        pos_encode_type="abs",
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            enc_num_attention_heads, enc_hidden_size, dropout_attn_rate
        )
        self.feed_forward = PositionwiseFeedForward(
            enc_hidden_size, enc_intermediate_size, dropout_enc_rate
        )
        # TODO: set `eps` to 1e-5 (default)
        # TODO: rename to `norm_self_attn` and `norm_ff`
        self.norm1 = nn.LayerNorm(enc_hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(enc_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_enc_rate)

    def forward(self, x, mask, pos_emb=None):
        residual = x
        x = self.norm1(x)  # normalize before
        x_q = x
        x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        residual = x
        x = self.norm2(x)  # normalize before
        x = residual + self.dropout(self.feed_forward(x))

        return x, mask


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dec_num_attention_heads,
        dec_hidden_size,
        dec_intermediate_size,
        dropout_dec_rate,
        dropout_attn_rate,
        pos_encode_type="abs",
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.dec_hidden_size = dec_hidden_size
        self.self_attn = MultiHeadedAttention(
            dec_num_attention_heads, dec_hidden_size, dropout_attn_rate
        )
        self.src_attn = MultiHeadedAttention(
            dec_num_attention_heads, dec_hidden_size, dropout_attn_rate
        )
        self.feed_forward = PositionwiseFeedForward(
            dec_hidden_size, dec_intermediate_size, dropout_dec_rate
        )
        # TODO: set `eps` to 1e-5 (default)
        self.norm1 = nn.LayerNorm(dec_hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(dec_hidden_size, eps=1e-12)
        self.norm3 = nn.LayerNorm(dec_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_dec_rate)

    def forward(self, x, mask, memory, memory_mask):
        residual = x
        x = self.norm1(x)  # normalize before
        x_q = x
        x_q_mask = mask
        x = residual + self.dropout(self.self_attn(x_q, x, x, x_q_mask))

        residual = x
        x = self.norm2(x)  # normalize before
        x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))

        residual = x
        x = self.norm3(x)  # normalize before
        x = residual + self.dropout(self.feed_forward(x))

        return x, mask, memory, memory_mask
