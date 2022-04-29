# Non-autoregressive Error Correction for CTC-based ASR with Phone-conditioned Masked LM

## Requirements

* Python 3.8

```
torch==1.7.1
gitpython==3.1.24
numpy==1.20.3
pandas==1.3.3
```

## Train ASR

```
python asr/train_asr.py -conf asr/correct/exps/csj/asr.yaml
```

## Train PC-MLM (Error Correction model)

```
python lm/train_lm.py -conf asr/correct/exps/csj/del_pc_mlm.yaml
```

## Test ASR

(without correction)
```
python asr/test_asr.py -conf asr/correct/exps/csj/asr.yaml -ep 91-100
```

(with correction)
```
python asr/test_asr_correct.py -conf asr/correct/exps/csj/asr.yaml -ep 91-100 -lm_conf asr/correct/exps/csj/del_pc_mlm.yaml -lm_ep 100 --lm_weight 0.5 --mask_th 0.8
```
