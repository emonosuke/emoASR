import os
import sys

import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from asr.models.model_utils import make_nopad_mask

from lm.models.transformers.configuration_transformers import TransformersConfig
from lm.models.transformers.modeling_bert import BertForMaskedLM


class TransformerLM(nn.Module):
    def __init__(self, params):
        super(TransformerLM, self).__init__()
        config = TransformersConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
            num_hidden_layers=params.num_layers,
            num_attention_heads=params.num_attention_heads,
            intermediate_size=params.intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.transformer = BertForMaskedLM(config)

    def forward(self, ys, ylens=None, labels=None):
        if ylens is None:
            attention_mask = None
        else:
            attention_mask = make_nopad_mask(ylens).float().to(ys.device)
            ys = ys[:, : max(ylens)]  # DataParallel

        loss = None
        loss_dict = {}
        # NOTE: causal attention mask
        if labels is None:
            (logits,) = self.transformer(ys, attention_mask=attention_mask, causal=True)
            return logits

        loss, logits = self.transformer(
            ys, attention_mask=attention_mask, causal=True, labels=labels
        )
        loss_dict["loss_total"] = loss

        return loss, loss_dict, logits

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.transformer.load_state_dict(state_dict)
