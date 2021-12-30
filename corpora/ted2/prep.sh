cd ../

# download dataset
wget http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz -P data/orig
tar xzf TEDLIUM_release2.tar.gz

# install sph2pipe
wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
tar xzf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5/

# sph -> wav
mkdir ted2/data/orig/TEDLIUM_release2/train/wav
for set in "train" "dev" "test"; do
    wavdir=ted2/data/${set}/wav
    mkdir -p $wavdir
    sphpaths="ted2/data/orig/TEDLIUM_release2/${set}/sph/*.sph"
    for sphpath in $sphpaths; do
        wavpath=${sphpath//sph/wav}
        ted2/sph2pipe_v2.5/sph2pipe -f wav -p $sphpath $wavpath
        echo "${sphpath} -> ${wavpath}"
    done
done

# speed perturbation
wavpaths="ted2/data/orig/TEDLIUM_release2/train/wav/*.wav"
mkdir ted2/data/orig/TEDLIUM_release2/train/wav_sp
for speed in "0.9" "1.0" "1.1"; do
    for wavpath in $wavpaths; do
        wav=$(basename ${wavpath})
        spwavpath="ted2/data/orig/TEDLIUM_release2/train/wav_sp/sp${speed}-${wav}"
        sox ${wavpath} ${spwavpath} speed ${speed}
        echo "${wavpath} -> ${spwavpath}"
    done
done
for set in "dev" "test"; do
    mkdir ted2/data/orig/TEDLIUM_release2/${set}/wav_sp
    cp ted2/data/orig/TEDLIUM_release2/${set}/wav/*.wav ted2/data/orig/TEDLIUM_release2/${set}/wav_sp/.
done

# split wav for utterances
for set in "train" "dev" "test"; do
    stmdir=ted2/data/orig/TEDLIUM_release2/${set}/stm
    wavdir=ted2/data/orig/TEDLIUM_release2/${set}/wav_sp
    outwavdir=ted2/data/${set}/feats
    mkdir $outwavdir
    tsvpath=ted2/data/${set}_feats.tsv
    if [${set} = "train"]; then
        python ted2/make_utts.py $stmdir $wavdir $outwavdir $tsvpath --extend_time --speed_perturb
    else
        python ted2/make_utts.py $stmdir $wavdir $outwavdir $tsvpath
    fi
done

for set in "train" "dev" "test"; do
    # skip `ignore_time_segment_in_scoring`
    python utils/rm_utt.py ted2/data/${set}_feats.tsv
    # e.g. it 's -> it's
    python ted2/join_suffix.py ted2/data/${set}_feats.tsv
done

# wav -> lmfb (npy)
for set in "train" "dev" "test"; do
    python utils/wav_to_feats.py ted2/data/${set}_feats.tsv
done

# normalize by train
for set in "train" "dev" "test"; do
    python utils/norm_feats.py ted2/data/${set}_feats.tsv ted2/data/train_feats_norm.pkl
done

# tokenize
mkdir ted2/data/sp10k
python utils/get_cols.py ted2/data/train_feats.tsv -cols text --no_header -out ted2/data/train_feats.txt
python utils/spm_train.py ted2/data/train_feats.txt -model ted2/data/sp10k/sp10k.model -vocab ted2/data/sp10k/vocab.txt -vocab_size 10000
for set in "train" "dev" "test"; do
    python utils/spm_encode.py ted2/data/${set}_feats.tsv -model ted2/data/sp10k/sp10k.model -vocab ted2/data/sp10k/vocab.txt --out ted2/data/sp10k/${set}.tsv
    python ted2/prep_tsv.py ted2/data/sp10k/${set}.tsv
done

python utils/sort_bylen.py ted2/data/sp10k/train.tsv
