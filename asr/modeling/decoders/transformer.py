""" Transformer
"""

import os
import sys
from operator import itemgetter

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../")
sys.path.append(EMOASR_ROOT)

from asr.criteria import DistillLoss, LabelSmoothingLoss
from asr.modeling.decoders.ctc import CTCDecoder
from asr.modeling.model_utils import make_src_mask, make_tgt_mask
from asr.modeling.transformer import PositionalEncoder, TransformerDecoderLayer
from lm.criteria import MaskedLMLoss
from utils.converters import strip_eos


class TransformerDecoder(nn.Module):
    def __init__(self, params, cmlm=False):
        super(TransformerDecoder, self).__init__()

        self.vocab_size = params.vocab_size
        self.embed = nn.Embedding(self.vocab_size, params.dec_hidden_size)
        self.pe = PositionalEncoder(params.dec_hidden_size, params.dropout_dec_rate)
        self.dec_num_layers = params.dec_num_layers

        # TODO: rename to `decoders`
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
        # TODO: set `eps` to 1e-5 (default)
        self.norm = nn.LayerNorm(params.dec_hidden_size, eps=1e-12)
        self.output = nn.Linear(params.dec_hidden_size, self.vocab_size)

        self.cmlm = cmlm
        if self.cmlm:
            # TODO: label smoothing
            self.loss_fn = MaskedLMLoss(vocab_size=self.vocab_size)
        else:
            self.loss_fn = LabelSmoothingLoss(
                vocab_size=self.vocab_size,
                lsm_prob=params.lsm_prob,
                normalize_length=params.loss_normalize_length,
                normalize_batch=params.loss_normalize_batch,
            )

        self.kd_weight = params.kd_weight
        if self.kd_weight > 0:
            self.loss_fn = DistillLoss(
                vocab_size=self.vocab_size,
                soft_label_weight=params.kd_weight,
                lsm_prob=params.lsm_prob,
                normalize_length=params.loss_normalize_length,
                normalize_batch=params.loss_normalize_batch,
            )

        self.eos_id = params.eos_id
        self.max_decode_ylen = params.max_decode_ylen

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

        # embedding + positional encoding
        ys_in = self.pe(self.embed(ys_in))
        emask = make_src_mask(elens)

        if self.cmlm:  # Conditional Masked LM
            ymask = make_src_mask(ylens)
        else:
            ymask = make_tgt_mask(ylens + 1)

        for layer_id in range(self.dec_num_layers):
            ys_in, ymask, eouts, emask = self.transformers[layer_id](
                ys_in, ymask, eouts, emask
            )
        ys_in = self.norm(ys_in)  # normalize before
        logits = self.output(ys_in)

        if ys_out is None:
            return logits

        if self.kd_weight > 0 and soft_labels is not None:
            # NOTE: ys_out (label) have length ylens+1
            loss_att_kd, loss_kd, loss_att = self.loss_fn(
                logits, ys_out, soft_labels, ylens + 1
            )

            loss += loss_att_kd
            loss_dict["loss_kd"] = loss_kd
            loss_dict["loss_att"] = loss_att
        else:
            if self.cmlm:
                loss_att = self.loss_fn(logits, labels=ys_out, ylens=None)
            else:
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

    def forward_one_step(self, ys_in, ylens_in, eouts):
        ys_in = self.pe(self.embed(ys_in))
        ymask = make_tgt_mask(ylens_in)

        for layer_id in range(self.dec_num_layers):
            ys_in, ymask, eouts, _ = self.transformers[layer_id](
                ys_in, ymask, eouts, None
            )

        ys_in = self.norm(ys_in[:, -1])  # normalize before
        logits = self.output(ys_in)
        return logits

    def decode(
        self,
        eouts,
        elens,
        eouts_inter=None,
        beam_width=1,
        len_weight=0,
        decode_ctc_weight=0,
        lm=None,
        lm_weight=0,
        decode_phone=False,
    ):
        """ Beam search decoding
        """
        bs = eouts.size(0)
        if decode_ctc_weight == 1:
            print("CTC is used")
            # greedy
            return self.ctc.decode(eouts, elens, beam_width=1)

        assert bs == 1

        # init
        beam = {"hyp": [self.eos_id], "score": 0.0, "lm_state": None}
        beams = [beam]

        results = []

        for i in range(self.max_decode_ylen):
            new_beams = []

            for beam in beams:
                ys_in = torch.tensor([beam["hyp"]]).to(eouts.device)
                ylens_in = torch.tensor([i + 1]).to(eouts.device)

                scores_asr = torch.log_softmax(
                    self.forward_one_step(ys_in, ylens_in, eouts), dim=-1
                )  # (1, vocab)
                scores = scores_asr

                if lm_weight > 0:
                    scores_lm, _ = lm.predict(ys_in, state=None)
                    scores += lm_weight * scores_lm[: self.vocab_size]

                scores_topk, v_topk = torch.topk(scores, k=beam_width, dim=1)

                for j in range(beam_width):
                    new_beam = {}
                    new_beam["score"] = beam["score"] + float(scores_topk[0, j])
                    new_beam["hyp"] = beam["hyp"] + [int(v_topk[0, j])]
                    new_beams.append(new_beam)

            # update `beams`
            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[
                :beam_width
            ]

            beams_extend = []
            for beam in beams:
                # ended beams
                if beam["hyp"][-1] == self.eos_id:
                    hyp_noeos = strip_eos(beam["hyp"], self.eos_id)
                    # only <eos> is not acceptable
                    if len(hyp_noeos) < 1:
                        continue

                    # add length penalty
                    score = beam["score"] + len_weight * len(beam["hyp"])

                    results.append({"hyp": hyp_noeos, "score": score})

                    if len(results) >= beam_width:
                        break
                else:
                    beams_extend.append(beam)

            if len(results) >= beam_width:
                break

            beams = beams_extend

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        hyps = [result["hyp"] for result in results]
        scores = [result["score"] for result in results]
        logits = None
        aligns = None

        return hyps, scores, logits, aligns
