# emoASR

## Features

## Results

### Librispeech

| |  | params | clean | other |
|:---:|:---|:---:|:---:|:---:|
| `L1` | CTC(Trf.) | 20M | 5.5 | 12.7 |
| `L2` | CTC(Cf.) | 23M | 4.2 | 10.1 |
| `L3` | Trf.(Cf.) | 35M | 3.2 | 7.0 |
| `L3-1` | +CTC |  |  |
| `L3-2` | +SF |  |  |
| `L3-3` | +CTC+SF |  |  |
| `L4` | Transducer(Cf.) 1kBPE | 26M |  |  |

### TED-LIUM2

|  |  | test | dev |
|:---:|:---|:---:|:---:|
| `T1` | CTC(Trf.) | 11.8 | 12.4 |
| `T2` | CTC(Cf.) | 10.1 | 10.8 |
| `T3` | Trf.(Cf.) | 7.8 | 11.5 |
| `T3-1` | +CTC |  |  |
| `T3-2` | +SF |  |  |
| `T3-3` | +CTC+SF |  |  |
| `T4` | Transducer(Trf.) 1kBPE | 9.5 | 10.5 |
| `T5` | Transducer(Cf.) 1kBPE |  |  |

### CSJ[WER/CER]

|  |  | eval1 | eval2 | eval3 |
|:---:|:---|:---:|:---:|:---:|
| `C1` | CTC(Trf.) | 8.4/6.4 | 6.4/5.0 | 7.0/5.1 |
| `C2` | Trf.(Trf.) | 6.7/5.0 | 4.9/3.6 | 5.5/4.0 |
| `C3` | Trf.(Cf.) |  |

## Reference

