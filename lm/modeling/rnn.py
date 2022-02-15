import os
import sys

import torch
import torch.nn as nn

EMOASR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_DIR)

from utils.converters import tensor2np


class RNNLM(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.embed = nn.Embedding(params.vocab_size, params.embedding_size)
        self.rnns = nn.LSTM(
            input_size=params.embedding_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            dropout=params.dropout_rate,
            batch_first=True,
        )
        self.output = nn.Linear(params.hidden_size, params.vocab_size)
        self.dropout = nn.Dropout(params.dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()  # ignore_index = -100
        
        self.num_layers = params.num_layers
        self.hidden_size = params.hidden_size
        self.vocab_size = params.vocab_size

        if params.tie_weights:
            pass
    
    def forward(self, ys, ylens=None, labels=None, ps=None, plens=None):
        if ylens is not None:
            # DataParallel
            ys = ys[:, : max(ylens)]
        
        ys_emb = self.dropout(self.embed(ys))
        out, _ = self.rnns(ys_emb)
        logits = self.output(self.dropout(out))

        if labels is None:
            return logits
        
        if ylens is not None:
            labels = labels[:, : max(ylens)]
        loss = self.loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))
        loss_dict = {"loss_total": loss}

        return loss, loss_dict
    
    def zero_states(self, bs, device):
        zeros = torch.zeros(
            self.num_layers, bs, self.hidden_size, device=device
        )
        # hidden state, cell state
        return (zeros, zeros)
    
    def predict(self, ys, ylens, states=None):
        """ predict next token for Shallow Fusion
        """
        ys_last = []
        bs = len(ys)
        for b in range(bs):
            ys_last.append(tensor2np(ys[b, ylens[b] - 1 : ylens[b]]))
        ys_last = torch.tensor(ys_last).to(ys.device)

        #print("ys:", ys)
        #print("ylens:", ylens)
        #print("ys_last:", ys_last)
        #print("ys_last:", ys_last.shape)
        ys_last_emb = self.dropout(self.embed(ys_last))
        out, states = self.rnns(ys_last_emb, states)
        logits = self.output(self.dropout(out))
        #print("logits:", logits.shape)
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs[:, -1], states

    def score(self, ys, ylens, batch_size=None):
        """ score token sequence for Rescoring
        """
        pass
