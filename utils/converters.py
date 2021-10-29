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


def get_ys_in_out(ys, ylens, eos_id, add_sos_eos=False):
    if add_sos_eos:
        eos = torch.tensor([eos_id], dtype=ys.dtype, device=ys.device)

        ys_in_list = [torch.cat((eos, y)) for y, ylen in zip(ys, ylens)]
        ys_out_list = [torch.cat((y, eos)) for y, ylen in zip(ys, ylens)]
    else:
        ys_in_list = [y[0 : (ylen - 1)] for y, ylen in zip(ys, ylens)]
        ys_out_list = [y[1:ylen] for y, ylen in zip(ys, ylens)]

    ys_in = pad_sequence(ys_in_list, batch_first=True, padding_value=eos_id)
    ys_out = pad_sequence(ys_out_list, batch_first=True, padding_value=eos_id)

    return ys_in, ys_out


def tensor2np(x):
    return x.cpu().detach().numpy()


def np2tensor(array, device=None):
    return torch.from_numpy(array).to(device)
