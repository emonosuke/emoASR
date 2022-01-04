import os
import sys

import numpy as np
import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from asr.modeling.model_utils import make_nopad_mask
from utils.converters import tensor2np

from lm.modeling.transformers.configuration_transformers import \
    TransformersConfig
from lm.modeling.transformers.modeling_bert import BertForMaskedLM


class TransformerLM(nn.Module):
    def __init__(self, params):
        super().__init__()
        config = TransformersConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
            num_hidden_layers=params.num_layers,
            num_attention_heads=params.num_attention_heads,
            intermediate_size=params.intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.transformer = BertForMaskedLM(config)

        # if params.tie_weights:
        #     pass

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is None:
            attention_mask = None
        else:
            attention_mask = make_nopad_mask(ylens).float().to(ys.device)
            # DataParallel
            ys = ys[:, : max(ylens)]
        
        if labels is None:
            # NOTE: causal attention mask
            (logits,) = self.transformer(ys, attention_mask=attention_mask, causal=True)
            return logits
        
        if ylens is not None:
            labels = labels[:, : max(ylens)]
        # NOTE: causal attention mask
        loss, logits = self.transformer(
            ys, attention_mask=attention_mask, causal=True, labels=labels
        )
        loss_dict = {"loss_total": loss}

        return loss, loss_dict

    def predict(self, ys, ylens, states=None):
        """ predict next token for Shallow Fusion
        """
        attention_mask = make_nopad_mask(ylens).float().to(ys.device)

        with torch.no_grad():
            (logits,) = self.transformer(ys, attention_mask, causal=True)

        log_probs = torch.log_softmax(logits, dim=-1)

        log_probs_next = []
        bs = len(ys)
        for b in range(bs):
            log_probs_next.append(tensor2np(log_probs[b, ylens[b] - 1]))

        return torch.tensor(log_probs_next).to(ys.device), states

    def score(self, ys, ylens, batch_size=None):
        """ score token sequence for Rescoring
        """
        attention_mask = make_nopad_mask(ylens).float().to(ys.device)

        with torch.no_grad():
            (logits,) = self.transformer(ys, attention_mask, causal=True)

        log_probs = torch.log_softmax(logits, dim=-1)

        score_lms = []
        bs = len(ys)
        for b in range(bs):
            score_lm = 0

            for i in range(0, ylens[b] - 1):
                v = ys[b, i + 1].item()  # predict next
                score_lm += log_probs[b, i, v].item()
            score_lms.append(score_lm)

        return score_lms

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.transformer.load_state_dict(state_dict)
