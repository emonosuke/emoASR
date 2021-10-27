import argparse
import json
import logging
import os
import re
import sys

import torch

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.paths import get_model_path


def model_average(conf: str, ep: str):
    paths = []

    if "-" in ep:
        startep = int(ep.split("-")[0])
        endep = int(ep.split("-")[1])
        epochs = [epoch for epoch in range(startep, endep + 1)]
    elif "+" in ep:
        epochs = list(map(int, ep.split("+")))
    else:
        return
    logging.info(f"average checkpoints... (epoch: {epochs})")

    for epoch in epochs:
        paths.append(get_model_path(conf, str(epoch)))

    save_path = re.sub("model.ep[0-9]+", f"model.ep{ep}", paths[0])
    if os.path.exists(save_path):
        logging.info(f"checkpoint: {save_path} already exists!")
        return

    avg = None
    # sum
    for path in paths:
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            avg[k] = torch.div(avg[k], len(paths))

    torch.save(avg, save_path)
    logging.info(f"checkpoints saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("-ep", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )

    model_average(args.conf, args.ep)
