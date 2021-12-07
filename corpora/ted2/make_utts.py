import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def main(args):
    rows = []  # utt_id, wav_path, text

    for stm_file in tqdm(sorted(os.listdir(args.stm_dir))):
        stm_path = os.path.join(args.stm_dir, stm_file)
        if not stm_path.endswith(".stm"):
            continue

        # read stm
        with open(stm_path) as f:
            lines = f.readlines()
        for line in lines:
            sections = line.strip().split()
            utt_prefix = sections[0]
            wav_path = os.path.join(args.wav_dir, f"{utt_prefix}.wav")
            start_time = float(sections[3])
            end_time = float(sections[4])
            text = " ".join(sections[6:])
            start_time_str = str(int(start_time * 100)).zfill(7)
            end_time_str = str(int(end_time * 100)).zfill(7)
            utt_id = f"{utt_prefix}-{start_time_str}-{end_time_str}"

            out_wav_dir = os.path.join(args.wav_dir, utt_prefix)
            os.makedirs(out_wav_dir, exist_ok=True)
            out_wav_path = os.path.join(out_wav_dir, f"{utt_id}.wav")

            # trim wav
            cp = subprocess.run(
                [
                    "sox",
                    wav_path,
                    out_wav_path,
                    "trim",
                    f"{start_time:.2f}",
                    f"={end_time:.2f}",
                ]
            )
            assert cp.returncode == 0

            rows.append((utt_id, out_wav_path, text))

    data = pd.DataFrame(rows, columns=["utt_id", "wav_path", "text"])
    data.to_csv(args.tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stm_dir", type=str)
    parser.add_argument("wav_dir", type=str)
    parser.add_argument("tsv_path", type=str)
    args = parser.parse_args()

    main(args)
