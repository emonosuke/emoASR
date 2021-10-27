import argparse
import gc
import os
import sys

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
sys.path.append(EMOASR_ROOT)

import pandas as pd
from tqdm import tqdm

from utils.converters import ints2str, str2ints


def main(args):
    data = pd.read_table(args.tsv_path)
    # data = data.dropna()

    print(f"Read tsv ({len(data)} samples)")

    # shuffle
    if args.shuffle:
        data = data.sample(frac=1, random_state=0).reset_index(drop=True)
        print(f"Data shuffled")
    else:
        print(f"Data NOT shuffled")

    # concat sentences (its lengths is NOT always the same as args.max_len)
    if args.task == "P2W":
        utt_id_start, utt_id_end = "", ""
        phone_token_id_concat = [args.phone_eos_id]
        phone_text_concat = "<eos>"
        token_id_concat = [args.eos_id]
        text_concat = "<eos>"

        outs = []  # utt_id, phone_token_id, phone_text, token_id, text

        for row in tqdm(data.itertuples()):
            utt_id = row.utt_id
            phone_token_id = str2ints(row.phone_token_id) + [args.phone_eos_id]
            token_id = str2ints(row.token_id) + [args.eos_id]
            phone_text = f" {row.phone_text} <eos>"
            text = f" {row.text} <eos>"

            if len(phone_token_id) + 1 > args.max_src_len:
                continue
            if len(token_id) + 1 > args.max_len:
                continue

            if utt_id_start == "":
                utt_id_start = row.utt_id
            utt_id_end = row.utt_id

            # NOTE: filter by its length
            if (
                len(phone_token_id_concat) + len(phone_token_id) > args.max_src_len
                or len(token_id_concat) + len(token_id) > args.max_len
            ):
                if (
                    len(phone_token_id_concat) >= args.min_src_len
                    and len(token_id_concat) >= args.min_len
                ):
                    outs.append(
                        (
                            f"{utt_id_start}-{utt_id_end}",
                            ints2str(phone_token_id_concat),
                            phone_text_concat,
                            ints2str(token_id_concat),
                            text_concat,
                        )
                    )

                utt_id_start, utt_id_end = "", ""
                phone_token_id_concat = [args.phone_eos_id]
                phone_text_concat = "<eos>"
                token_id_concat = [args.eos_id]
                text_concat = "<eos>"

            else:
                phone_token_id_concat.extend(phone_token_id)
                token_id_concat.extend(token_id)
                phone_text_concat += phone_text
                text_concat += text

        if utt_id_start != "":
            if (
                len(phone_token_id_concat) >= args.min_src_len
                and len(token_id_concat) >= args.min_len
            ):
                outs.append(
                    (
                        f"{utt_id_start}-{utt_id_end}",
                        ints2str(phone_token_id_concat),
                        phone_text_concat,
                        ints2str(token_id_concat),
                        text_concat,
                    )
                )
        data = pd.DataFrame(
            outs,
            columns=["utt_id", "phone_token_id", "phone_text", "token_id", "text",],
        )

    # concat tokens (its lengths is always the same as args.max_len)
    # NOTE: sentence longer than max_len is skipped
    elif args.task == "LM":
        utt_id_start, utt_id_end = "", ""
        token_id_concat = [args.eos_id]

        outs = []  # utt_id, token_id, text

        for row in tqdm(data.itertuples()):
            utt_id = row.utt_id
            token_id = str2ints(row.token_id) + [args.eos_id]

            if utt_id_start == "":
                utt_id_start = row.utt_id
            utt_id_end = row.utt_id

            if len(token_id) > args.max_len:
                continue

            if len(token_id_concat) + len(token_id) < args.max_len:
                token_id_concat += token_id
            else:
                remainder = args.max_len - len(token_id_concat)
                token_id_concat += token_id[:remainder]
                assert len(token_id_concat) == args.max_len
                outs.append((f"{utt_id_start}-{utt_id_end}", ints2str(token_id_concat)))
                utt_id_start, utt_id_end = "", ""
                token_id_concat = token_id[remainder:]

        # NOTE: text cannot provide
        data = pd.DataFrame(outs, columns=["utt_id", "token_id"],)

    elif args.task == "LMall":
        if args.eos_id >= 0:
            token_id_all = [args.eos_id]
        else:
            token_id_all = []

        # NOTE: First, concat all tokens
        for row in data.itertuples():
            token_id_all.extend(str2ints(row.token_id))
            if args.eos_id >= 0:
                token_id_all.append(args.eos_id)

        # save memory
        del data
        gc.collect()

        start = 0
        utt_id_prefix = os.path.splitext(os.path.basename(args.tsv_path))[0]
        outs = []  # utt_id, token_id

        for i in range(args.rep):
            start = 0 + i * (args.max_len // args.rep)
            while start + args.max_len < len(token_id_all):
                end = start + args.max_len
                outs.append(
                    (f"{utt_id_prefix}-{i}-{start}", ints2str(token_id_all[start:end]))
                )
                start = end

        # NOTE: text cannot provide
        data = pd.DataFrame(outs, columns=["utt_id", "token_id"],)

    if args.out is None:
        data.to_csv(
            f"{os.path.splitext(args.tsv_path)[0]}_concat.tsv", sep="\t", index=False
        )
    else:
        data.to_csv(args.out, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("-task", choices=["P2W", "LM", "LMall"], required=True)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--min_len", type=int, default=64)
    # NOTE: max source length is set to 1024
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--min_src_len", type=int, default=64)
    parser.add_argument("--eos_id", type=int, default=2)
    parser.add_argument("--phone_eos_id", type=int, default=2)
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()
    main(args)
