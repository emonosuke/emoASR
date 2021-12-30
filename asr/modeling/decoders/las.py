""" LAS (Listen Attend Spell)

Reference:
    https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/models/seq2seq/decoders/las.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import DistillLoss, LabelSmoothingLoss
from asr.modeling.decoders.ctc import CTCDecoder
from asr.modeling.model_utils import make_nopad_mask


class LASDecoder(nn.Module):
    def __init__(self, params, phase="train"):
        super().__init__()

        self.enc_hidden_size = params.enc_hidden_size
        self.dec_hidden_size = params.dec_hidden_size
        self.dec_num_layers = params.dec_num_layers
        self.mtl_ctc_weight = params.mtl_ctc_weight
        if self.mtl_ctc_weight > 0:
            self.ctc = CTCDecoder(params)

        self.embed = nn.Embedding(params.vocab_size, params.embedding_size)
        self.dropout_emb = nn.Dropout(p=params.dropout_dec_rate)

        # Recurrency
        self.rnns = nn.ModuleList()
        input_size = params.embedding_size + params.enc_hidden_size
        for _ in range(self.dec_num_layers):
            self.rnns += [
                nn.LSTMCell(
                    input_size, params.dec_hidden_size,
                )
            ]
            input_size = params.dec_hidden_size

        # Score
        self.score = AttentionLoc(key_dim=params.enc_hidden_size, query_dim=params.dec_hidden_size, attn_dim=params.attn_dim)

        # Generate
        self.intermed = nn.Linear(params.enc_hidden_size + params.dec_hidden_size, params.dec_intermediate_size)
        self.output = nn.Linear(params.dec_intermediate_size, params.vocab_size)

        self.dropout = nn.Dropout(p=params.dropout_dec_rate)

        self.loss_fn = LabelSmoothingLoss(
            vocab_size=params.vocab_size,
            lsm_prob=params.lsm_prob,
            normalize_length=params.loss_normalize_length,
            normalize_batch=params.loss_normalize_batch,
        )

        self.kd_weight = params.kd_weight
        if self.kd_weight > 0:
            pass

    def forward(
        self,
        eouts,
        elens,
        eouts_inter=None,
        ys=None,
        ylens=None,
        ys_in=None,
        ys_out=None,  # labels
        soft_labels=None,
        ps=None,
        plens=None,
    ):
        loss = 0
        loss_dict = {}

        bs = eouts.size(0)
        
        ys_emb = self.dropout_emb(self.embed(ys_in))
        dstate = None
        ctx = eouts.new_zeros(bs, 1, self.enc_hidden_size)
        attn_weight = None
        attn_mask = make_nopad_mask(elens).unsqueeze(2)
        logits = []
        
        for i in range(ys_in.size(1)):
            y_emb = ys_emb[:, i : i + 1]

            # Recurrency -> Score -> Generate
            dstate, douts_1, douts_top = self.recurrency(torch.cat([y_emb, ctx], dim=-1), dstate)
            ctx, attn_weight = self.score(eouts, eouts, douts_1, attn_weight, attn_mask)
            logit = self.generate(ctx, douts_top)

            logits.append(logit)
        
        logits = self.output(torch.cat(logits, dim=1))  # (bs, ylen, vocab)

        # NOTE: ys_out (label) have length ylens+1
        loss_att = self.loss_fn(logits, ys_out, ylens + 1)
        loss += loss_att
        loss_dict["loss_att"] = loss_att

        if self.mtl_ctc_weight > 0:
            # NOTE: KD is not applied to auxiliary CTC
            loss_ctc, _, _ = self.ctc(
                eouts=eouts, elens=elens, ys=ys, ylens=ylens, soft_labels=None
            )
            loss += self.mtl_ctc_weight * loss_ctc  # auxiliary loss
            loss_dict["loss_ctc"] = loss_ctc

        loss_dict["loss_total"] = loss

        return loss, loss_dict, logits

    def recurrency(self, dins, dstate=None):
        bs = dins.size(0)
        douts = dins.squeeze(1)

        if dstate is None:
            dstate = {}
            dstate["hs"] = torch.zeros(
                self.dec_num_layers, bs, self.dec_hidden_size, device=dins.device
            )
            dstate["cs"] = torch.zeros(
                self.dec_num_layers, bs, self.dec_hidden_size, device=dins.device
            )

        new_hs, new_cs = [], []
        for layer_id in range(self.dec_num_layers):
            h, c = self.rnns[layer_id](douts, (dstate["hs"][layer_id], dstate["cs"][layer_id]))
            new_hs.append(h)
            new_cs.append(c)
            douts = self.dropout(h)
            if layer_id == 0:
                douts_1 = douts.unsqueeze(1)

        new_dstate = {}
        new_dstate["hs"] = torch.stack(new_hs, dim=0)
        new_dstate["cs"] = torch.stack(new_cs, dim=0)

        douts_top = douts.unsqueeze(1)

        return new_dstate, douts_1, douts_top

    def generate(self, ctx, douts):
        out = self.intermed(torch.cat([ctx, douts], dim=-1))
        
        return torch.tanh(out)

class AttentionLoc(nn.Module):
    def __init__(self, key_dim, query_dim, attn_dim, conv_out_channels=10, conv_kernel_size=201, dropout_attn_rate=0.1):
        super().__init__()

        self.w_key = nn.Linear(key_dim, attn_dim)
        self.w_query = nn.Linear(query_dim, attn_dim)
        self.w_conv = nn.Linear(conv_out_channels, attn_dim)
        self.w_score = nn.Linear(attn_dim, 1)

        assert conv_kernel_size % 2 == 1
        # (N, 1, L_in) -> (N, conv_out_channels, L_in)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=(conv_kernel_size - 1) // 2,
            bias=False
        )

        self.dropout = nn.Dropout(p=dropout_attn_rate)

    def forward(self, key, value, query, attn_weight=None, attn_mask=None):
        """
        key: (bs, klen, key_dim)
        value: (bs, klen, key_dim)
        query: (bs, 1, query_dim)
        """
        bs = key.size(0)
        klen = key.size(1)

        if attn_weight is None:
            attn_weight = key.new_zeros(bs, 1, klen)

        conv_feat = self.conv(attn_weight)  # (bs, channel, klen)
        conv_feat = conv_feat.transpose(1, 2)  # (bs, klen, channel)
        score = self.w_score(
            torch.tanh(
                self.w_key(key) + self.w_query(query) + self.w_conv(conv_feat)
            )  # (bs, klen, attn_dim)
        )  # (bs, klen, 1)

        if attn_mask is not None:
            NEG_INF = float(
                np.finfo(torch.tensor(0, dtype=score.dtype).numpy().dtype).min
            )
            score = score.masked_fill_(attn_mask == 0, NEG_INF)

        attn_weight = torch.softmax(score, dim=1)
        attn_weight = self.dropout(attn_weight)
        ctx = torch.sum(attn_weight * value, dim=1, keepdim=True)
        attn_weight = attn_weight.transpose(1, 2)

        return ctx, attn_weight
