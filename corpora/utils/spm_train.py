import argparse
import os

import sentencepiece as spm


def build_vocab(
    spm_vocab_path, vocab_path, special_tokens=["<pad>", "<unk>", "<eos>", "<pad>"]
):
    outs = []
    for i, token in enumerate(special_tokens):
        outs.append(f"{token} {i:d}\n")
    with open(spm_vocab_path) as f:
        for i, line in enumerate(f):
            token = line.split()[0]
            print(token)
            outs.append(f"{token} {(i+len(special_tokens)):d}\n")
    with open(vocab_path, "w") as f:
        f.writelines(outs)


def main(args):
    if not os.path.exists(args.model):
        spm.SentencePieceTrainer.train(
            input=args.data,
            model_prefix=args.model.replace(".model", ""),
            vocab_size=args.vocab_size,
        )
    build_vocab(args.model.replace(".model", ".vocab"), args.vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)  # .tsv
    parser.add_argument("-model", type=str, required=True)
    parser.add_argument("-vocab", type=str, required=True)
    parser.add_argument("-vocab_size", type=int, required=True)
    args = parser.parse_args()
    main(args)
