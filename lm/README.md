## Results

### Librispeech

[nsp10k]

| |  | params | PPL(clean) |
|:---:|:---|:---:|:---:|
| `L1` | Transformer | 12M | 63.9 |
| `L2` | BERT | 12M | 12.1 |
| `L3` | RNN | 13M | 75.2 |

- `L1`: `exps/libri_nsp10k/transformer`
- `L2`: `exps/libri_nsp10k/bert`
- `L3`: `exps/libri_nsp10k/rnn`

### TED-LIUM2

[nsp10k]

| |  | params | PPL(test) |
|:---:|:---|:---:|:---:|
| `T1` | Transformer | 12M | 56.3 |
| `T2` | BERT | 12M | 11.6 |
| `T3` | RNN | 13M | 69.6 |

- `T1`: `exps/ted2_nsp10k/transformer`
- `T2`: `exps/ted2_nsp10k/bert`
- `T3`: `exps/ted2_nsp10k/rnn`

### CSJ

[nsp10k]

| |  | params | PPL(eval1) |
|:---:|:---|:---:|:---:|
| `C1` | Transformer (BCCWJ init) | 25M | 53.7 |
| `C2` | BERT (BCCWJ init) | 25M | 8.6 |
| `S1` | Transformer (BCCWJ init) | 12M | 38.4 |
| `S2` | Transformer (w/o init) | 12M | 40.8 |
| `S3` | BERT (BCCWJ init) | 12M | 10.0 |
| `S4` | BERT (w/o init) | 12M | x |
| `S5` | RNN (BCCWJ init) | 14M | 39.5 |
| `S6` | RNN (w/o init) | 14M | 49.0 |

- `C1`: `exps/csj_nsp10k_bccwj/bccwj_csj_transformer`
- `C2`: `exps/csj_nsp10k_bccwj/bccwj_csj_bert`
- `S1`: `exps/csj_nsp10k_bccwj/bccwj_csj_transformer_small`
- `S2`: `exps/csj_nsp10k_bccwj/csj_transformer_small`
- `S3`: `exps/csj_nsp10k_bccwj/bccwj_csj_bert_small`
- `S4`: `exps/csj_nsp10k_bccwj/csj_bert_small`
- `S5`: `exps/csj_nsp10k_bccwj/bccwj_csj_rnn_small`
- `S6`: `exps/csj_nsp10k_bccwj/csj_rnn_small`
