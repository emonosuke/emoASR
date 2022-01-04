import torch.nn as nn


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

    def score(self, ys, ylens):
        """ score token sequence for Rescoring
        """
        pass
