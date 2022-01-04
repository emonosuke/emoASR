import os
import sys

import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from asr.modeling.model_utils import make_nopad_mask

from lm.modeling.transformers.configuration_transformers import \
    TransformersConfig
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

        # if params.tie_weights:
        #     pass

        self.mask_id = params.mask_id

    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is None:
            attention_mask = None
        else:
            attention_mask = make_nopad_mask(ylens).float().to(ys.device)
            # DataParallel
            ys = ys[:, : max(ylens)]

        if labels is None:
            (logits,) = self.bert(ys, attention_mask=attention_mask)
            return logits

        if ylens is not None:
            labels = labels[:, : max(ylens)]
        loss, logits = self.bert(ys, attention_mask=attention_mask, labels=labels)
        loss_dict = {"loss_total": loss}

        return loss, loss_dict
    
    def score(self, ys, ylens, batch_size=100):
        """ score token sequence for Rescoring
        """
        score_lms = []

        for y, ylen in zip(ys, ylens):
            ys_masked, mask_pos, mask_label = [], [], []

            score_lm = 0

            for pos in range(ylen):
                y_masked = y[:ylen].clone()
                y_masked[pos] = self.mask_id
                ys_masked.append(y_masked)
                mask_pos.append(pos)
                mask_label.append(y[pos])

                if len(ys_masked) < batch_size and pos != (ylen - 1):
                    continue
                
                ys_masked = torch.stack(ys_masked, dim=0)
                (logits,) = self.bert(ys_masked)
                logprobs = torch.log_softmax(logits, dim=-1)

                bs = ys_masked.size(0)
                for b in range(bs):
                    score_lm += logprobs[b, mask_pos[b], mask_label[b]].item()

                ys_masked, mask_pos, mask_label = [], [], []

            score_lms.append(score_lm)
        
        return score_lms

    def load_state_dict(self, state_dict):
        try:
            super().load_state_dict(state_dict)
        except:
            self.bert.load_state_dict(state_dict)
