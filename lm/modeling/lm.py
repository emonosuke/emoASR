""" Language modeling
"""

import logging
import os
import sys

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from utils.log import get_num_parameters

from lm.modeling.bert import BERTMaskedLM
from lm.modeling.electra import ELECTRAModel, PELECTRAModel
from lm.modeling.rnn import RNNLM
from lm.modeling.transformer import TransformerLM


class LM(nn.Module):
    def __init__(self, params, phase="train"):
        super().__init__()

        self.lm_type = params.lm_type
        logging.info(f"LM type: {self.lm_type}")

        if self.lm_type == "bert":
            self.lm = BERTMaskedLM(params)
        elif self.lm_type == "transformer":
            self.lm = TransformerLM(params)
        elif self.lm_type in ["electra", "electra-disc"]:
            self.lm = ELECTRAModel(params)
        elif self.lm_type in ["pelectra", "pelectra-disc"]:
            self.lm = PELECTRAModel(params)
        elif self.lm_type == "rnn":
            self.lm = RNNLM(params)

        num_params, num_params_trainable = get_num_parameters(self)
        logging.info(
            f"LM model #parameters: {num_params} ({num_params_trainable} trainable)"
        )

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        return self.lm(ys, ylens, labels, ps, plens)

    def forward_disc(self, ys, ylens, error_labels):
        return self.lm.forward_disc(ys, ylens, error_labels)
    
    def zero_states(self, bs, device):
        return self.lm.zero_states(bs, device)

    def predict(self, ys, ylens, states=None):
        with torch.no_grad():
            return self.lm.predict(ys, ylens, states)

    def score(self, ys, ylens, batch_size=100):
        with torch.no_grad():
            return self.lm.score(ys, ylens, batch_size)

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.lm.load_state_dict(state_dict)
