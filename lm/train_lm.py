""" train LM
"""
import argparse
import datetime
import logging
import math
import os
import random
import socket
import sys
import time

import git
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(EMOASR_ROOT)

from asr.optimizers import (ScheduledOptimizer, get_optimizer_params_nodecay,
                            optimizer_to)
from utils.configure import load_config
from utils.paths import (get_log_save_paths, get_model_optim_paths,
                         rel_to_abs_path)

from lm.datasets import LMBatchSampler, LMDataset, P2WDataset
from lm.modeling.lm import LM
from lm.modeling.p2w import P2W

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# disable warning from huggingface
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_step(
    model, optimizer, data, params, device, no_grad=False, empty_cache=False
):
    ys_in = data["ys_in"].to(device)
    ylens = data["ylens"].to(device)
    labels = data["labels"].to(device)

    # for P2WDataset
    ps = data["ps"].to(device) if "ps" in data else None
    plens = data["plens"].to(device) if "plens" in data else None

    loss, loss_dict = model(ys_in, ylens, labels, ps, plens)

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


def valid(model, params, device, epoch):
    pass


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
    logging.info(f"torch: {torch.__version__}")
    commit_hash = git.Repo(search_parent_directories=True).head.object.hexsha
    logging.info(f"commit: {commit_hash}")
    logging.info(f"conda env: {os.environ['CONDA_DEFAULT_ENV']}")
    logging.info(f"torch version: {torch.__version__}")

    if params.lm_type in ["ptransformer", "pbert", "pctc"]:
        model = P2W(params)
    else:
        model = LM(params)

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

    num_total_steps = (
        params.train_size // (params.batch_size * params.accum_grad)
    ) * params.num_epochs
    logging.info(f"#steps: {num_total_steps:d}")

    optimizer_params = get_optimizer_params_nodecay(
        list(model.named_parameters()), weight_decay=params.weight_decay
    )
    # Adam with weight decay fix
    optimizer_base = AdamW(optimizer_params, lr=0, weight_decay=params.weight_decay)
    optimizer = ScheduledOptimizer(
        optimizer_base, params, num_total_steps=num_total_steps
    )
    if optim_path:
        optimizer.load_state_dict(torch.load(optim_path, map_location=device))
        logging.info(f"optimizer: {optim_path}")
    else:
        logging.info(f"optimizer: scratch")

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
    train_path = rel_to_abs_path(params.train_path)

    for epoch in range(startep, params.num_epochs):
        _time = time.time()
        # training data is splitted to files
        if os.path.isdir(train_path):
            train_files = os.listdir(train_path)
            random.shuffle(train_files)
            for step_ds, train_file in enumerate(train_files):
                train_file_path = os.path.join(train_path, train_file)

                if params.lm_type in ["pelectra", "ptransformer", "pbert", "pctc"]:
                    dataset = P2WDataset(params, train_file_path)
                else:
                    dataset = LMDataset(params, train_file_path)

                if params.bucket_shuffle:
                    dataloader = DataLoader(
                        dataset=dataset,
                        batch_sampler=LMBatchSampler(
                            dataset, params, min_batch_size=num_gpus
                        ),
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
                        shuffle=True,
                        num_workers=1,
                        pin_memory=True,
                        collate_fn=dataset.collate_fn,
                    )
                logging.info(
                    f"Dataset ({(step_ds+1):d}/{len(train_files):d}): {train_file_path}"
                )
                train(model, optimizer, dataloader, params, device, epoch)
        else:
            if params.lm_type in ["pelectra", "ptransformer", "pbert", "pctc"]:
                dataset = P2WDataset(params, train_path)
            else:
                dataset = LMDataset(params, train_path)

            if params.bucket_shuffle:
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_sampler=LMBatchSampler(
                        dataset, params, min_batch_size=num_gpus
                    ),
                    collate_fn=dataset.collate_fn,
                    num_workers=1,
                )
                logging.info(
                    f"{len(dataset):d} samples -> {len(dataloader):d} batches (batch size average: {(len(dataset)/len(dataloader)):.2f})"
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn,
                )
            logging.info(f"Dataset: {train_path}")
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
