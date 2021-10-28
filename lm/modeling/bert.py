import os
import sys

import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from asr.modeling.model_utils import make_nopad_mask
from utils.io_utils import load_config

from lm.modeling.transformers.configuration_transformers import TransformersConfig
from lm.modeling.transformers.modeling_bert import BertForMaskedLM


class BERTMaskedLM(nn.Module):
    def __init__(self, params):
        super(BERTMaskedLM, self).__init__()
        config = TransformersConfig(
            vocab_size=params.vocab_size,
            hidden_size=params.hidden_size,
            num_hidden_layers=params.num_layers,
            num_attention_heads=params.num_attention_heads,
            intermediate_size=params.intermediate_size,
            max_position_embeddings=params.max_seq_len,
        )
        self.bert = BertForMaskedLM(config)

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is None:
            attention_mask = None
        else:
            attention_mask = make_nopad_mask(ylens).float().to(ys.device)
            # DataParallel
            ys = ys[:, : max(ylens)]
            labels = labels[:, : max(ylens)]

        loss = None
        loss_dict = {}
        if labels is None:
            (logits,) = self.bert(ys, attention_mask=attention_mask)
            return logits

        loss, logits = self.bert(ys, attention_mask=attention_mask, labels=labels)
        loss_dict["loss_total"] = loss

        return loss, loss_dict

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.bert.load_state_dict(state_dict)
