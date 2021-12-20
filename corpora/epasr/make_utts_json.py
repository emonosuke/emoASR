import argparse
import json
import os
import subprocess

import pandas as pd


def main(args):
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
                    if file.endswith(args.json_ext):
                        json_path = os.path.join(data_dir3, file)
                assert wav_path and json_path

                wav_file = os.path.basename(wav_path)
                assert "ep-asr.en.orig." in wav_file
                wav_file = wav_file.replace("ep-asr.en.orig.", "")
                utt_prefix = wav_file.replace(".wav", "")

                out_wav_dir = os.path.join(args.out_wav_dir, utt_prefix)
                os.makedirs(out_wav_dir, exist_ok=True)

                with open(json_path) as f:
                    sections = json.load(f)
                for section in sections:
                    start_time = float(section["b"])
                    end_time = float(section["e"])
                    start_time_str = str(int(start_time * 100)).zfill(7)
                    end_time_str = str(int(end_time * 100)).zfill(7)
                    text = " ".join([sec["w"] for sec in section["wl"]])
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
    parser.add_argument("data_dir", type=str)  # wav + json
    # `train`
    parser.add_argument("out_wav_dir", type=str)
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("json_ext", type=str)
    args = parser.parse_args()

    main(args)
