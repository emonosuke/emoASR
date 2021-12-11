# emoASR

## Features

### ASR modeling

* Encoder
    * RNN
    * Transformer
    * Conformer
* Decoder
    * CTC
    * Transformer
    * RNN-Transducer

## Results

### Librispeech[WER]

| |  | params | clean | other |
|:---:|:---|:---:|:---:|:---:|
| `L1` | CTC(Trf.) | 20M | 5.5* | 12.7* |
| `L2` | CTC(Cf.) | 23M | 4.2 | 10.1 |
| `L3` | Trf.(Cf.) | 35M | 3.2 | 7.0 |
| `L3-1` | +CTC | - | 2.9 | 6.9 |
| `L3-2` | +SF | - | 2.9 | 6.3 |
| `L3-3` | +CTC+SF | - | **2.5** | **6.0** |
| `L4` | RNN-T(Cf.) 1kBPE | 26M | 2.9 | 7.1 |

### TED-LIUM2[WER]

|  |  | params | test | dev |
|:---:|:---|:---:|:---:|:---:|
| `T1` | CTC(Trf.) | 20M | 10.9 | 12.4 |
| `T2` | CTC(Cf.) | 23M | 9.4 | 10.1 |
| `T3` | Trf.(Cf.) | 35M | 7.8 | 11.5 |
| `T3-1` | +CTC | - | 7.4 | 9.6 |
| `T3-2` | +SF | - | 7.4 | 10.7 |
| `T3-3` | +CTC+SF | - | **6.8** | 9.2 |
| `T4` | RNN-T(Trf.) 1kBPE | 22M | 9.5 | 10.5 |
| `T5` | RNN-T(Cf.) 1kBPE | 26M | 7.4 | **8.3** |

### CSJ[WER/CER]

|  |  | params | eval1 | eval2 | eval3 |
|:---:|:---|:---:|:---:|:---:|:---:|
| `C1` | CTC(Trf.) | 20M | 8.1/6.2 | 6.4/4.8 | 6.9/5.0 |
| `C2` | CTC(Cf.) | 24M | 6.8/5.0 | 5.3/4.0 | 5.9/4.3 |
| `C3` | Trf.(Trf.) | 32M | 6.7/5.0 | **4.9**/3.6 | 5.5/4.0 |
| `C4` | Trf.(Cf.) | 36M | **6.3**/4.7 | 5.0/3.8 | **5.2**/4.0 |

## Reference

