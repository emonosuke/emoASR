# Rescoring

"ASR RESCORING AND CONFIDENCE ESTIMATION WITH ELECTRA" [Futami ASRU2021]  
https://arxiv.org/pdf/2110.01857.pdf

### TED-LIUM2

| |  | WER(test) | Runtime(test) |
|:---:|:---|:---:|:---:|
| `T1` | CTC ASR (w/o LM) | 12.11 | - |
| `T2` | +Transformer | 9.78 | x1 |
| `T3` | +BERT | 9.72 |  |
| `T4` | +ELECTRA | 10.17 |  |
| `T5` | +ELECTRA(FT) | ... |  |
| `T6` | +P-ELECTRA | 9.73 |  |
| `T7` | +P-ELECTRA(FT) | ... |  |

- `T1`: `exps/ted2_nsp10k/transformer_ctc`
- `T2`: `exps/ted2_nsp10k/transformer`
- `T3`: `exps/ted2_nsp10k/bert`
- `T4`: `exps/ted2_nsp10k/electra_utt`
- `T5`: `exps/ted2_nsp10k/electra_utt_asr5b`
- `T6`: `exps/ted2_nsp10k/pelectra_utt`
- `T7`: `exps/ted2_nsp10k/pelectra_utt_asr5b`
