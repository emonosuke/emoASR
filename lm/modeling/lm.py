import logging
import os
import sys

import torch
import torch.nn as nn

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

from lm.modeling.bert import BERTMaskedLM
from lm.modeling.electra import ELECTRAModel, PELECTRAModel
from lm.modeling.transformerlm import TransformerLM


class LM(nn.Module):
    def __init__(self, params, phase="train"):
        super(LM, self).__init__()

        self.lm_type = params.lm_type
        logging.info(f"LM type: {self.lm_type}")

        if self.lm_type == "bert":
            self.lm = BERTMaskedLM(params)
        elif self.lm_type == "transformer":
            self.lm = TransformerLM(params)
        elif self.lm_type == "electra":
            self.lm = ELECTRAModel(params)
        elif self.lm_type == "pelectra":
            self.lm = PELECTRAModel(params)

        num_params = sum(p.numel() for p in self.parameters())
        num_params_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logging.info(
            f"LM model #parameters: {num_params} ({num_params_trainable} trainable)"
        )

    def forward(self, ys, ylens, labels):
        return self.lm(ys, ylens, labels)

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.lm.load_state_dict(state_dict)
