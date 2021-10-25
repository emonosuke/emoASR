""" train ASR
"""
import argparse
import datetime
import logging
import math
import os
import socket
import sys
import time

import git
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from utils.configure import load_config
from utils.paths import get_log_save_paths, get_model_optim_paths, rel_to_abs_path
from utils.vocab import Vocab

from asr.datasets import ASRBatchSampler, ASRDataset
from asr.metrics import compute_wers
from asr.modeling.asr import ASR
from asr.optimizers import ScheduledOptimizer, optimizer_to

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def train_step(
    model, optimizer, data, params, device, no_grad=False, empty_cache=False
):
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    ys = data["ys"].to(device)
    ylens = data["ylens"].to(device)
    ys_in = data["ys_in"].to(device)
    ys_out = data["ys_out"].to(device)

    if "soft_labels" in data:
        soft_labels = data["soft_labels"].to(device)
    else:
        soft_labels = None

    loss, loss_dict = model(
        xs, xlens, ys, ylens, ys_in, ys_out, soft_labels=soft_labels
    )

    # reduction for devices
    if torch.cuda.device_count() > 1:
        loss = loss.mean()
        loss_dict = {
            loss_key: loss_value.mean() for loss_key, loss_value in loss_dict.items()
        }

    loss_dict = {
        loss_key: loss_value.item() / params.accum_grad
        for loss_key, loss_value in loss_dict.items()
    }
    loss /= params.accum_grad

    loss.backward()

    # skip when accumulating gradients
    if not no_grad:
        # gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), params.clip_grad_norm
        )

        if math.isnan(grad_norm):
            logging.warning("do not update because of nan grad_norm")
        else:
            optimizer.step()
        optimizer.zero_grad()

        if empty_cache:
            torch.cuda.empty_cache()

    return loss_dict


def train(model, optimizer, dataloader, params, device, epoch):
    optimizer.update_epoch()

    step = 0
    loss_dict_sum = {}

    for accum_step, data in enumerate(dataloader):
        if (accum_step + 1) % params.accum_grad == 0:
            loss_dict = train_step(
                model,
                optimizer,
                data,
                params,
                device,
                no_grad=False,
                empty_cache=args.empty_cache,
            )
            step += 1
        else:
            loss_dict = train_step(
                model, optimizer, data, params, device, no_grad=True, empty_cache=False
            )

        if not loss_dict_sum:
            for loss_key in loss_dict.keys():
                loss_dict_sum[loss_key] = loss_dict[loss_key]
        else:
            for loss_key in loss_dict.keys():
                loss_dict_sum[loss_key] += loss_dict[loss_key]

        if step % params.log_step == 0 and (accum_step + 1) % params.accum_grad == 0:
            loss_detail = " ".join(
                [
                    f"{loss_key}: {loss_value/params.log_step:.3f}"
                    for loss_key, loss_value in loss_dict_sum.items()
                ]
            )

            logging.info(
                f"epoch = {(epoch+1):>2} step = {step:>6} / {len(dataloader)//params.accum_grad:>6} lr = {optimizer._lr:.5f} "
                + loss_detail
            )

            loss_dict_sum = {}


def valid_step(model, data, device):
    xs = data["xs"].to(device)
    xlens = data["xlens"].to(device)
    ys = data["ys"]
    ylens = data["ylens"]

    if torch.cuda.device_count() > 1:
        hyps, _ = model.module.decode(xs, xlens, beam_width=0, len_weight=0)
    else:
        hyps, _ = model.decode(xs, xlens, beam_width=0, len_weight=0)

    refs = [y[:ylen].tolist() for y, ylen in zip(ys, ylens)]

    return hyps, refs


def valid(model, params, device, epoch):
    dev_size = params.dev_size if hasattr(params, "dev_size") else -1
    vocab = Vocab(vocab_path=rel_to_abs_path(params.vocab_path), size=dev_size)
    dataset_val = ASRDataset(params, rel_to_abs_path(params.dev_path), phase="valid")
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=params.batch_size,
        collate_fn=dataset_val.collate_fn,
        num_workers=1,
    )
    logging.info(f"dev data: {params.dev_path} ({len(dataloader_val)} steps)")

    hyps_val, refs_val = [], []
    for data in dataloader_val:
        hyps, refs = valid_step(model, data, device)
        hyps_val.extend(hyps)
        refs_val.extend(refs)
    assert len(hyps_val) == len(refs_val)

    wer_val, _ = compute_wers(hyps_val, refs_val, vocab)
    logging.info(f"*** epoch = {(epoch+1):d}: valid WER = {wer_val:.2f}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = load_config(args.conf)

    log_dir, save_format, optim_save_format = get_log_save_paths(args.conf)

    if args.debug:
        logging.basicConfig(
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            filename=os.path.join(log_dir, "train.log"),
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            level=logging.INFO,
        )

    logging.info(f"***** {' '.join(sys.argv)}")
    logging.info(
        f"server: {socket.gethostname()} | gpu: {os.getenv('CUDA_VISIBLE_DEVICES')} | pid: {os.getpid():d}"
    )
    commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    logging.info(f"commit: {commit_hash}")
    logging.info(f"conda env: {os.environ['CONDA_DEFAULT_ENV']}")
    logging.info(f"torch version: {torch.__version__}")

    model = ASR(params)
    model_path, optim_path, startep = get_model_optim_paths(
        args.conf,
        resume=args.resume,
        model_path=params.model_path,
        optim_path=params.optim_path,
        start_epoch=params.startep,
    )
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"model: {model_path}")
    else:
        logging.info(f"model: scratch")

    optimizer_base = Adam(model.parameters(), lr=0, weight_decay=params.weight_decay)
    optimizer = ScheduledOptimizer(optimizer_base, params)
    if optim_path:
        optimizer.load_state_dict(torch.load(optim_path, map_location=device))
        logging.info(f"optimizer: {optim_path}")
    else:
        logging.info(f"optimizer: scratch")

    # DataParallel
    if torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        device_ids = list(range(num_gpus))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        num_gpus = 1
    model.to(device)
    model.train()
    optimizer_to(optimizer, device)

    logging.info(f"train data: {params.train_path}")
    dataset = ASRDataset(params, rel_to_abs_path(params.train_path), phase="train")

    if params.train_data_shuffle:
        # bucket shuffling
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=ASRBatchSampler(dataset, params, min_batch_size=num_gpus),
            collate_fn=dataset.collate_fn,
            num_workers=1,
        )
        logging.info(
            f"{len(dataset):d} samples -> {len(dataloader):d} batches (batch size average: {(len(dataset)/len(dataloader)):.2f})"
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            num_workers=1,
        )

    for epoch in range(startep, params.num_epochs):
        _time = time.time()
        train(model, optimizer, dataloader, params, device, epoch)
        elapsed_time = datetime.timedelta(seconds=(time.time() - _time))
        end_time = datetime.datetime.now() + elapsed_time * (
            params.num_epochs - (epoch + 1)
        )
        logging.info(f"epoch = {(epoch+1):>2} elapsed time: {elapsed_time}")
        logging.info(f"time to end: {end_time}")

        logging.info("validation start")
        model.eval()
        try:
            valid(model, params, device, epoch)
        except:
            logging.error("ERROR occurs in validation (ignore)", exc_info=True)
        logging.info("validation end")
        # be sure to be training mode
        model.train()

        if epoch == 0 or (epoch + 1) % params.save_step == 0:
            if args.debug:
                continue

            save_path = save_format.format(epoch + 1)
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            optim_save_path = optim_save_format.format(epoch + 1)
            torch.save(optimizer.state_dict(), optim_save_path)
            logging.info(f"model saved to: {save_path}")
            logging.info(f"optimizer saved to: {optim_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--empty_cache", action="store_true")  # unrecommended
    args = parser.parse_args()

    try:
        main(args)
    except:
        logging.error("***** ERROR occurs in training *****", exc_info=True)
        logging.error("**********")
