import codecs
import logging
import os
import re
from collections import namedtuple

import yaml

EMOASR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")


def get_eval_path(ref_tag):
    # ted2
    if ref_tag == "test":  # TODO: -> ted2-test
        return os.path.join(EMOASR_ROOT, "corpora/ted2/nsp10k/data/test.tsv")
    if ref_tag == "dev":  # TODO: -> ted2-dev
        return os.path.join(EMOASR_ROOT, "corpora/ted2/nsp10k/data/dev.tsv")

    # libri
    if ref_tag == "test-clean":  # TODO: -> libri-test-clean
        return os.path.join(EMOASR_ROOT, "corpora/libri/nsp10k/data/test_clean.tsv")
    if ref_tag == "test-other":  # TODO: -> libri-test-other
        return os.path.join(EMOASR_ROOT, "corpora/libri/nsp10k/data/test_other.tsv")
    if ref_tag == "dev-clean":  # TODO: -> libri-dev-clean
        return os.path.join(EMOASR_ROOT, "corpora/libri/nsp10k/data/dev_clean.tsv")
    if ref_tag == "dev-other":  # TODO: -> libri-dev-other
        return os.path.join(EMOASR_ROOT, "corpora/libri/nsp10k/data/dev_other.tsv")

    # csj
    if ref_tag == "eval1":  # TODO: -> csj-eval1
        return os.path.join(EMOASR_ROOT, "corpora/csj/nsp10k/data/eval1.tsv")
    elif ref_tag == "eval2":
        return os.path.join(EMOASR_ROOT, "corpora/csj/nsp10k/data/eval2.tsv")
    elif ref_tag == "eval3":
        return os.path.join(EMOASR_ROOT, "corpora/csj/nsp10k/data/eval3.tsv")
    elif ref_tag == "csj-dev":
        return os.path.join(EMOASR_ROOT, "corpora/csj/nsp10k/data/dev.tsv")
    elif ref_tag == "csj-dev500":
        return os.path.join(EMOASR_ROOT, "corpora/csj/nsp10k/data/dev_500.tsv")

    return ref_tag


def get_run_dir(conf_path):
    run_dir = os.path.splitext(conf_path)[0]
    return run_dir


def get_exp_dir(conf_path):
    exp_dir = os.path.splitext(conf_path)[0]
    return exp_dir


def get_model_path(conf_path, epoch):
    run_dir = get_run_dir(conf_path)
    model_dir = os.path.join(run_dir, "checkpoints")
    model_path = os.path.join(model_dir, f"model.ep{epoch}")
    return model_path


def get_results_dir(conf_path):
    run_dir = get_run_dir(conf_path)
    results_dir = os.path.join(run_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_log_save_paths(conf_path):
    run_dir = get_run_dir(conf_path)
    os.makedirs(run_dir, exist_ok=True)
    log_dir = os.path.join(run_dir, "log")
    save_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    save_format = os.path.join(save_dir, "model.ep{}")
    optim_save_format = os.path.join(save_dir, "optim.ep{}")

    return log_dir, save_format, optim_save_format


def get_resume_paths(conf_path, epoch=0):
    run_dir = get_run_dir(conf_path)
    save_dir = os.path.join(run_dir, "checkpoints")

    model_ep_max = 0
    optim_ep_max = 0

    if epoch > 0:
        model_path = os.path.join(save_dir, f"model.ep{epoch:d}")
        optim_path = os.path.join(save_dir, f"optim.ep{epoch:d}")
    # if epoch is not given, find latest model and optim
    else:
        for ckpt_file in os.listdir(save_dir):
            match = re.fullmatch(r"model.ep([0-9]+)", ckpt_file)
            if match is not None:
                model_ep = int(match.group(1))
                model_ep_max = max(model_ep, model_ep_max)

            match = re.fullmatch(r"optim.ep([0-9]+)", ckpt_file)
            if match is not None:
                optim_ep = int(match.group(1))
                optim_ep_max = max(optim_ep, optim_ep_max)

        assert model_ep_max == optim_ep_max
        epoch = model_ep_max

        if epoch > 0:
            model_path = os.path.join(save_dir, f"model.ep{epoch:d}")
            optim_path = os.path.join(save_dir, f"optim.ep{epoch:d}")
        else:
            model_path, optim_path = "", ""

    return model_path, optim_path, epoch


def get_model_optim_paths(
    conf_path, resume=False, model_path=None, optim_path=None, start_epoch=0
):
    resume_model_path, resume_optim_path, resume_epoch = "", "", 0
    if resume:
        resume_model_path, resume_optim_path, resume_epoch = get_resume_paths(conf_path)
        if resume_epoch > 0:
            logging.info(f"resume from epoch = {resume_epoch:d}")

    model_path = resume_model_path or model_path
    optim_path = resume_optim_path or optim_path
    start_epoch = resume_epoch or start_epoch

    return model_path, optim_path, start_epoch


def rel_to_abs_path(path):
    if os.path.exists(path):
        return path
    else:
        return os.path.join(EMOASR_ROOT, path)
