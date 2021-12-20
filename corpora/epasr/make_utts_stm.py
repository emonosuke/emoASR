import argparse
import json
import os
import subprocess

import pandas as pd


def main(args):
    stm_labels = {}
    with open(args.stm_path) as f:
        lines = f.readlines()
    for line in lines:
        sections = line.strip().split()
        utt_prefix = sections[0].replace("ep-asr.en.orig.", "")
        start_time = float(sections[3])
        end_time = float(sections[4])
        text = " ".join(sections[6:])
        if utt_prefix not in stm_labels:
            stm_labels[utt_prefix] = [(start_time, end_time, text)]
        else:
            stm_labels[utt_prefix].append((start_time, end_time, text))

    rows = []  # utt_id, wav_path, text

    for data_file in sorted(os.listdir(args.data_dir)):  # e.g.`t6`
        data_dir1 = os.path.join(args.data_dir, data_file)
        for data_file in sorted(os.listdir(data_dir1)):  # e.g.`2009-04-21`
            data_dir2 = os.path.join(data_dir1, data_file)
            for data_file in sorted(os.listdir(data_dir2)):  # e.g.`2-196`
                data_dir3 = os.path.join(data_dir2, data_file)
                files = [file for file in os.listdir(data_dir3)]
                wav_path, json_path = "", ""
                for file in files:
                    if file.endswith(".wav"):
                        wav_path = os.path.join(data_dir3, file)

                wav_file = os.path.basename(wav_path)
                assert "ep-asr.en.orig." in wav_file
                wav_file = wav_file.replace("ep-asr.en.orig.", "")
                utt_prefix = wav_file.replace(".wav", "")

                out_wav_dir = os.path.join(args.out_wav_dir, utt_prefix)
                os.makedirs(out_wav_dir, exist_ok=True)

                for start_time, end_time, text in stm_labels[utt_prefix]:
                    start_time_str = str(int(start_time * 100)).zfill(7)
                    end_time_str = str(int(end_time * 100)).zfill(7)
                    utt_id = f"{utt_prefix}-{start_time_str}-{end_time_str}"

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

                    print(f"{wav_path} -> {out_wav_path}")

                    rows.append((utt_id, out_wav_path, text))

    data = pd.DataFrame(rows, columns=["utt_id", "wav_path", "text"])
    data.to_csv(args.tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)  # wav
    # `dev-dep`, `dev-indep`, `test-dep`, `test-indep`
    parser.add_argument("out_wav_dir", type=str)
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("stm_path", type=str)
    args = parser.parse_args()

    main(args)
