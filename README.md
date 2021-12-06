# emoASR

## Features

## Results

### Librispeech[WER]

| |  | params | clean | other |
|:---:|:---|:---:|:---:|:---:|
| `L1` | CTC(Trf.) | 20M | 5.5* | 12.7* |
| `L2` | CTC(Cf.) | 23M | 4.2* | 10.1* |
| `L3` | Trf.(Cf.) | 35M | 3.2 | 7.0 |
| `L3-1` | +CTC | - | 2.9 | 6.9 |
| `L3-2` | +SF | - | * | * |
| `L3-3` | +CTC+SF | - | * | * |
| `L4` | Transducer(Cf.) 1kBPE | 26M | 3.3* | 7.9* |

### TED-LIUM2[WER]

|  |  | params | test | dev |
|:---:|:---|:---:|:---:|:---:|
| `T1` | CTC(Trf.) | 20M | 11.8* | 12.4* |
| `T2` | CTC(Cf.) | 23M | 10.1* | 10.8* |
| `T3` | Trf.(Cf.) | 35M | 7.8 | 11.5 |
| `T3-1` | +CTC | - | 7.4 | 9.6 |
| `T3-2` | +SF | - | * | * |
| `T3-3` | +CTC+SF | - | * | * |
| `T4` | Transducer(Trf.) 1kBPE | 22M | 9.5 | 10.5 |
| `T5` | Transducer(Cf.) 1kBPE | 26M | * | * |

### CSJ[WER/CER]

|  |  | params | eval1 | eval2 | eval3 |
|:---:|:---|:---:|:---:|:---:|:---:|
| `C1` | CTC(Trf.) | 20M | 8.4/6.4* | 6.4/5.0* | 7.0/5.1* |
| `C2` | CTC(Cf.) | 24M | 6.8/5.0 | 5.3/4.0 | 5.9/4.3 |
| `C3` | Trf.(Trf.) | 32M | 6.7/5.0 | 4.9/3.6 | 5.5/4.0 |
| `C4` | Trf.(Cf.) | 36M | * | * | * |

## Reference

