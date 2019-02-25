# download
mkdir data
mkdir data_vectors
mkdir models
mkdir outputs

# download datasets and word vectors

wget http://59.108.48.35/aaai19_lattice/test_re_9413.json -O data/test_re_9413.json
wget http://59.108.48.35/aaai19_lattice/train_re_14262.json -O data/train_re_14262.json
wget http://59.108.48.35/aaai19_lattice/test_dbqa_5997.json -O data/test_dbqa_5997.json
wget http://59.108.48.35/aaai19_lattice/train_dbqa_8768.json -O data/train_dbqa_8768.json

wget http://59.108.48.35/aaai19_lattice/WordBank_v -O data_vectors/WordBank_v
wget http://59.108.48.35/aaai19_lattice/vectorsw300l20.all -O data_vectors/vectorsw300l20.all

# data preparing
cd src/data_preprocess
python3 get_nlpccdbqa.py
python3 get_nlpcckbqa.py





