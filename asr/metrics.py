"""

Reference
    - https://github.com/hirofumi0810/neural_sp/blob/master/neural_sp/evaluators/edit_distance.py
"""

import os
import sys

import numpy as np


def compute_wer(hyp, ref, cer=False):
    # NOTE: if cannot decode, use <dummy> symbol (never match with ref)
    if len(hyp) == 0:
        hyp = ["<dummy>"]

    if cer:
        hyp = list("".join(hyp))
        ref = list("".join(ref))

    # edit distance
    d = np.zeros((len(ref) + 1) * (len(hyp) + 1), dtype=np.uint16)
    d = d.reshape((len(ref) + 1, len(hyp) + 1))
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                sub_tmp = d[i - 1][j - 1] + 1
                ins_tmp = d[i][j - 1] + 1
                del_tmp = d[i - 1][j] + 1
                d[i][j] = min(sub_tmp, ins_tmp, del_tmp)
    dist = d[len(ref)][len(hyp)]

    # backtrack
    x = len(ref)
    y = len(hyp)
    error_list = []
    while True:
        if x == 0 and y == 0:
            break
        else:
            if x > 0 and y > 0:
                if d[x][y] == d[x - 1][y - 1] and ref[x - 1] == hyp[y - 1]:
                    error_list.append("C")
                    x = x - 1
                    y = y - 1
                elif d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                elif d[x][y] == d[x - 1][y - 1] + 1:
                    error_list.append("S")
                    x = x - 1
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif x == 0 and y > 0:
                if d[x][y] == d[x][y - 1] + 1:
                    error_list.append("I")
                    y = y - 1
                else:
                    error_list.append("D")
                    x = x - 1
            elif y == 0 and x > 0:
                error_list.append("D")
                x = x - 1
            else:
                raise ValueError
    error_list.reverse()

    n_sub = error_list.count("S")
    n_ins = error_list.count("I")
    n_del = error_list.count("D")
    n_cor = error_list.count("C")

    assert dist == (n_sub + n_ins + n_del)
    assert n_cor == (len(ref) - n_sub - n_del)

    wer = (dist / len(ref)) * 100
    wer_dict = {
        "wer": wer,
        "n_sub": n_sub,
        "n_ins": n_ins,
        "n_del": n_del,
        "n_ref": len(ref),
        "error_list": error_list,
    }

    return wer, wer_dict


def compute_wers(hyps, refs, vocab=None):
    n_sub_total, n_ins_total, n_del_total, n_ref_total = 0, 0, 0, 0

    for hyp, ref in zip(hyps, refs):
        if vocab is not None:
            hyp = vocab.ids2words(hyp)
            ref = vocab.ids2words(ref)

        _, wer_dict = compute_wer(hyp, ref)

        n_sub_total += wer_dict["n_sub"]
        n_ins_total += wer_dict["n_ins"]
        n_del_total += wer_dict["n_del"]
        n_ref_total += wer_dict["n_ref"]

    wer_total = ((n_sub_total + n_ins_total + n_del_total) / n_ref_total) * 100

    return wer_total


if __name__ == "__main__":
    hyp = ["ab", "cd", "efg", "h", "ij", "op"]
    ref = ["ab", "c", "defg", "h", "ij", "klm", "op"]

    print(compute_wer(hyp, ref))
    print(compute_wer(hyp, ref, cer=True))
