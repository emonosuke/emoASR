import argparse

import pandas as pd


def process_text(text):
    # it 's -> it's
    tokens = text.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i + 1][0] == "'":
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 1
        else:
            new_tokens.append(tokens[i])
        i += 1
    new_text = " ".join(new_tokens)
    return new_text


def main(args):
    df = pd.read_table(args.tsv_path)
    df["text"] = df["text"].map(process_text)
    df.to_csv(args.tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path", type=str)
    args = parser.parse_args()
    main(args)
