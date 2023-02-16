import torch
from torch import nn


class MaskedLMLoss(nn.Module):
    def __init__(self, vocab_size):
        super(MaskedLMLoss, self).__init__()

        # NOTE: pad with -100 (not masked)
        self.ce_loss = nn.CrossEntropyLoss()

        self.vocab_size = vocab_size

    def forward(self, logits, labels, ylens):
        loss = self.ce_loss(
            logits.contiguous().view(-1, self.vocab_size), labels.contiguous().view(-1),
        )
        return loss
