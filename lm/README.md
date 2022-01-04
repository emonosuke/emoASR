## Results

### Librispeech

[nsp10k]

| |  | params | PPL(clean) |
|:---:|:---|:---:|:---:|
| `T1` | Transformer | 12M | 63.9 |
| `T2` | BERT | 12M | 12.1 |
| `T3` | RNN | 13M | ... |

- `T1`: `exps/libri_nsp10k/transformer`
- `T2`: `exps/libri_nsp10k/bert`

### TED-LIUM2

[nsp10k]

| |  | params | PPL(test) |
|:---:|:---|:---:|:---:|
| `L1` | Transformer | 12M | 56.3 |
| `L2` | BERT | 12M | 11.6 |
| `L3` | RNN | 13M | ... |

- `L1`: `exps/ted2_nsp10k/transformer`
- `L2`: `exps/ted2_nsp10k/bert`

TODO: ELECTRA, P-ELECTRA

### CSJ

[nsp10k]

| |  | params | PPL(eval1) |
|:---:|:---|:---:|:---:|
| `C1` | Transformer (BCCWJ init) | 25M | 57.6 |
| `C2` | Transformer (w/o init) | 25M | 95.7 |
| `C3` | BERT (BCCWJ init) | 25M | 8.6 |
| `C4` | BERT (w/o init) | 25M | 9.3 |
| `C1` | Transformer (BCCWJ init) | 12M | ... |
| `C2` | Transformer (w/o init) | 12M | ... |
| `C3` | BERT (BCCWJ init) | 12M | ... |
| `C4` | BERT (w/o init) | 12M | ... |
| `C5` | RNN (BCCWJ init) | 12M | ... |
| `C6` | RNN (w/o init) | 12M | ... |

- `C1`: `exps/csj_nsp10k_bccwj/bccwj_csj_transformer_addeos`
- `C2`: `exps/csj_nsp10k_bccwj/csj_transformer_addeos`
- `C3`: `exps/csj_nsp10k_bccwj/bccwj_csj_bert`
- `C4`: `exps/csj_nsp10k_bccwj/csj_bert`
