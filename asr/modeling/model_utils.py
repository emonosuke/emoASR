import torch
from torch.nn.utils.rnn import pad_sequence


def make_nopad_mask(lengths):
    """
    NOTE: faster implementation of the following
    mask = [[bool(l < length) for l in range(max(lengths))] for length in lengths]
    """

    if torch.is_tensor(lengths):
        lens = lengths.tolist()
    else:
        lens = lengths

    bs = int(len(lengths))
    maxlen = int(max(lengths))

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lens).unsqueeze(-1)
    mask = seq_range_expand < seq_length_expand

    if torch.is_tensor(lengths):
        mask = mask.to(lengths.device)

    return mask


def make_causal_mask(length):
    ret = torch.ones(length, length, dtype=bool)
    return torch.tril(ret, out=ret)


def make_src_mask(xlens: torch.Tensor):
    return make_nopad_mask(xlens.tolist()).unsqueeze(-2).to(xlens.device)


def make_tgt_mask(ylens: torch.Tensor):
    nopad_mask = make_nopad_mask(ylens.tolist()).unsqueeze(-2)
    maxlen = nopad_mask.size(-1)
    causal_mask = make_causal_mask(maxlen).unsqueeze(0)
    return (nopad_mask & causal_mask).to(ylens.device)


if __name__ == "__main__":
    xlens = torch.tensor([1, 2, 3])
    src_mask = make_src_mask(xlens)
    tgt_mask = make_tgt_mask(xlens)
    print(src_mask)
    print(tgt_mask)
