# Non-autoregressive Error Correction for CTC-based ASR with Phone-conditioned Masked LM

## Requirements

* Python 3.8

```
torch==1.7.1
gitpython==3.1.24
numpy==1.20.3
pandas==1.3.3
```

## Data
Prepare train/test data in tsv format as follows:
```
utt_id  feat_path       xlen    token_id        text    ylen    phone_text      phone_token_id
S09F1170_0516169_0516727        path/to/S09F1170_0516169_0516727.npy 54      8 538 20        で なぜ か      3       d e n a z e k a 12 14 26 7 42 14 22 7
```
Note that `feat_path` and `xlen` is not required in PC-MLM training data.

## Train ASR
Train a Transformer-based CTC model on **SPS** subset of CSJ. It is trained for phone-level targets in addition to word-level ones, known as hierarchical multi-task learning.

```
python asr/train_asr.py -conf asr/correct/exps/csj/asr.yaml
```

Model checkpoints and log will be saved at `exps/csj/asr/`

## Train PC-MLM (Error Correction model)
Train a phone-conditioned masked LM (PC-MLM), which is a phone-to-word conversion model, on **APS** subset of CSJ.

Deletable version of PC-MLM (Del PC-MLM) that addresses insertion errors is trained as follows:
```
python lm/train_lm.py -conf asr/correct/exps/csj/del_pc_mlm.yaml
```

## Test ASR
Test ASR on `eval1` set for APS domain (domain adaptation setting).

Without correction (A1):
```
python asr/test_asr.py -conf asr/correct/exps/csj/asr.yaml -ep 91-100
```

With correction (A7):
```
python asr/test_asr_correct.py -conf asr/correct/exps/csj/asr.yaml -ep 91-100 -lm_conf asr/correct/exps/csj/del_pc_mlm.yaml -lm_ep 100 --lm_weight 0.5 --mask_th 0.8
```

Results are saved at `exps/csj/asr/results/`.
RTF can be calculated with `--runtime` option.

|  |  | WER | RTF |
|:---:|:---|:---:|:---:|
| (A1) | CTC (greedy) | 18.10 | 0.0033 |
| (A7) | +Correction (w/ Del PC-MLM) | 16.48 | 0.0094 |
