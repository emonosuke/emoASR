import torch


def str2ints(s):
    return list(map(int, s.split()))


def str2floats(s):
    return list(map(float, s.split()))


def ints2str(ints):
    return " ".join(list(map(str, ints)))


def get_utt_id_nosp(utt_id):
    if (
        utt_id.startswith("sp0.9")
        or utt_id.startswith("sp1.0")
        or utt_id.startswith("sp1.1")
    ):
        utt_id_nosp = "-".join(utt_id.split("-")[1:])
    else:
        utt_id_nosp = utt_id
    return utt_id_nosp


def strip_eos(tokens, eos_id):
    return [token for token in tokens if token != eos_id]


def tensor2np(x):
    return x.cpu().detach().numpy()


def np2tensor(array, device=None):
    return torch.from_numpy(array).to(device)
