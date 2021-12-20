cd ../

data=epasr/data/orig/release/en

train=${data}/train/original_audio/speeches
dev_dep=${data}/dev/original_audio/spk-dep/speeches; dev_indep=${data}/dev/original_audio/spk-indep/speeches
test_dep=${data}/test/original_audio/spk-dep/speeches; test_indep=${data}/test/original_audio/spk-indep/speeches

### ASR

for set in $train $dev_dep $dev_indep $test_dep $test_indep; do
    m4as=$(find $set -name "*.m4a")
    for m4a in $m4as; do
        wav=${m4a/.m4a/.wav}
        ffmpeg -y -i $m4a -ar 16000 $wav -loglevel error
        echo "${m4a} -> ${wav}"
    done
done

# split wav for utterances
python epasr/make_utts_json.py $train epasr/data/train epasr/data/train_wav.tsv ".tr.verb.json"

# read `stm`
python epasr/make_utts_stm.py $dev_dep epasr/data/dev_dep epasr/data/dev_dep_wav.tsv ${data}/dev/original_audio/spk-dep/refs/ep-asr.en.dev.spk-dep.rev.stm
python epasr/make_utts_stm.py $dev_indep epasr/data/dev_indep epasr/data/dev_indep_wav.tsv ${data}/dev/original_audio/spk-indep/refs/ep-asr.en.dev.spk-indep.rev.stm
python epasr/make_utts_stm.py $test_dep epasr/data/test_dep epasr/data/test_dep_wav.tsv ${data}/test/original_audio/spk-dep/refs/ep-asr.en.test.spk-dep.rev.stm
python epasr/make_utts_stm.py $test_indep epasr/data/test_indep epasr/data/test_indep_wav.tsv ${data}/test/original_audio/spk-indep/refs/ep-asr.en.test.spk-indep.rev.stm

# skip `ignore_time_segment_in_scoring`
for set in "dev_dep" "dev_indep" "test_dep" "test_indep"; do
    python utils/rm_utt.py epasr/data/${set}_wav.tsv
done

# wav -> lmfb (npy)
python utils/wav_to_feats.py epasr/data/train_wav.tsv
for set in "dev_dep" "dev_indep" "test_dep" "test_indep"; do
    python utils/wav_to_feats.py epasr/data/${set}_wav.tsv
done

# normalize
for set in "train" "dev_dep" "dev_indep" "test_dep" "test_indep"; do
    python utils/norm_feats.py epasr/data/${set}_wav.tsv epasr/data/train_wav_norm.pkl
done

# tokenize
for set in "train" "dev_dep" "dev_indep" "test_dep" "test_indep"; do
    python utils/spm_encode.py epasr/data/${set}_wav.tsv -model ted2/data/sp10k/sp10k.model -vocab ted2/data/sp10k/vocab.txt --out epasr/data/tedsp10k/${set}.tsv
    python ted2/prep_tsv.py epasr/data/tedsp10k/${set}.tsv
done

python utils/sort_bylen.py epasr/data/sp10k/train.tsv

### LM

