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
            start_time = float(sections[3])
            end_time = float(sections[4])
            text = " ".join(sections[6:])
            start_time_str = str(int(start_time * 100)).zfill(7)
            end_time_str = str(int(end_time * 100)).zfill(7)
            utt_id = f"{utt_prefix}-{start_time_str}-{end_time_str}"

            out_wav_dir = os.path.join(args.out_wav_dir, utt_prefix)
            os.makedirs(out_wav_dir, exist_ok=True)

            # The training set seems to not have enough silence padding in the segmentations,
            # especially at the beginning of segments.  Extend the times.
            if args.extend_time:
                start_time_fix = max(0, start_time - 0.15)
                end_time_fix = end_time + 0.1
            else:
                start_time_fix = start_time
                end_time_fix = end_time

            if args.speed_perturb:
                for speed in ["0.9", "1.0", "1.1"]:
                    wav_path = os.path.join(args.wav_dir, f"sp{speed}-{utt_prefix}.wav")
                    sp_utt_id = f"sp{speed}-{utt_id}"
                    out_wav_path = os.path.join(out_wav_dir, f"{sp_utt_id}.wav")
                    start_time_fix_sp = start_time_fix / float(speed)
                    end_time_fix_sp = end_time_fix / float(speed)

                    # trim wav
                    cp = subprocess.run(
                        [
                            "sox",
                            wav_path,
                            out_wav_path,
                            "trim",
                            f"{start_time_fix_sp:.2f}",
                            f"={end_time_fix_sp:.2f}",
                        ]
                    )
                    assert cp.returncode == 0
                    rows.append((sp_utt_id, out_wav_path, text))
            else:
                wav_path = os.path.join(args.wav_dir, f"{utt_prefix}.wav")
                out_wav_path = os.path.join(out_wav_dir, f"{utt_id}.wav")
                
                # trim wav
                cp = subprocess.run(
                    [
                        "sox",
                        wav_path,
                        out_wav_path,
                        "trim",
                        f"{start_time_fix:.2f}",
                        f"={end_time_fix:.2f}",
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
    parser.add_argument("out_wav_dir", type=str)
    parser.add_argument("tsv_path", type=str)
    parser.add_argument("--extend_time", action="store_true")
    parser.add_argument("--speed_perturb", action="store_true")
    args = parser.parse_args()

    main(args)
