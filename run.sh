# training
cd src

###  character level model

python3 run_nlpcc.py -m model_nlpcc_kbre_cws \
--train_base_func deal_nlpcckbqa_mws_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_mws --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segchar_real.json' --test '../data/test_re_9413_segchar_real.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_char' --output_path '../outputs/output_kbre_char'

python3 run_nlpcc.py -m model_nlpcc_irqa_cws \
--train_base_func deal_nlpccirqa_mws_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_mws --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segchar_real.json' --test '../data/test_dbqa_5997_segchar_real.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_char' --output_path '../outputs/output_irqa_char'

###  word level model, which used a fine tuned word segment method, result in finer granularity, less OOV words, and better performance. 
###  adopted in " NLPCC 2017 A chinese question answering system for single-relation factoid questions. "

python3 run_nlpcc.py -m model_nlpcc_kbre_word \
--train_base_func deal_nlpcckbqa_word_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_word --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segword.json' --test '../data/test_re_9413_segword.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_word' --output_path '../outputs/output_kbre_word'


python3 run_nlpcc.py -m model_nlpcc_irqa_word \
--train_base_func deal_nlpccirqa_word_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_word --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segword.json' --test '../data/test_dbqa_5997_segword.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_word' --output_path '../outputs/output_irqa_word'


###  

python3 run_nlpcc.py -m model_nlpcc_kbre_gcn \
--train_base_func deal_nlpcckbqa_gcn_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_gcn --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path ../data_vectors/vectorsw300l20.all \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_dgc' --output_path '../outputs/output_kbre_dgc' \
--model_param '{"layer_num": 1, "pooling_mode": "gcn", "if_d": true}'


python3 run_nlpcc.py -m model_nlpcc_irqa_gcn \
--train_base_func deal_nlpccirqa_gcn_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_gcn --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_dgc' --output_path '../outputs/output_irqa_dgc' \
--model_param '{"layer_num": 2, "pooling_mode": "gcn", "if_d": true}'

python3 run_nlpcc.py -m model_nlpcc_kbre_gcn \
--train_base_func deal_nlpcckbqa_gcn_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_gcn --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path ../data_vectors/vectorsw300l20.all \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_dgc_max' --output_path '../outputs/output_kbre_dgc_max' \
--model_param '{"layer_num": 1, "pooling_mode": "gcn_max", "if_d": true}'

python3 run_nlpcc.py -m model_nlpcc_irqa_gcn \
--train_base_func deal_nlpccirqa_gcn_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_gcn --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_dgc_max' --output_path '../outputs/output_irqa_dgc_max' \
--model_param '{"layer_num": 2, "pooling_mode": "gcn_max", "if_d": true}'

python3 run_nlpcc.py -m model_nlpcc_kbre_gcn \
--train_base_func deal_nlpcckbqa_gcn_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_gcn --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path ../data_vectors/vectorsw300l20.all \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_dgc_gated' --output_path '../outputs/output_kbre_dgc_gated' \
--model_param '{"layer_num": 1, "pooling_mode": "gat_n", "if_d": true}'

python3 run_nlpcc.py -m model_nlpcc_irqa_gcn \
--train_base_func deal_nlpccirqa_gcn_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_gcn --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_dgc_gated' --output_path '../outputs/output_irqa_dgc_gated' \
--model_param '{"layer_num": 2, "pooling_mode": "gat_n", "if_d": true}'


###  lattice based methods

python3 run_nlpcc.py -m model_nlpcc_kbre_cws \
--train_base_func deal_nlpcckbqa_mws_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_mws --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_lattice_max' --output_path '../outputs/output_kbre_lattice_max' \
--model_param '{"pooling_mode":"max", "layer_num":1}'

python3 run_nlpcc.py -m model_nlpcc_irqa_cws \
--train_base_func deal_nlpccirqa_mws_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_mws --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_lattice_max' --output_path '../outputs/output_irqa_lattice_max' \
--model_param '{"pooling_mode":"max", "layer_num":2}'

python3 run_nlpcc.py -m model_nlpcc_kbre_cws \
--train_base_func deal_nlpcckbqa_mws_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_mws --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_lattice_ave' --output_path '../outputs/output_kbre_lattice_ave' \
--model_param '{"pooling_mode":"ave", "layer_num":1}'

python3 run_nlpcc.py -m model_nlpcc_irqa_cws \
--train_base_func deal_nlpccirqa_mws_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_mws --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_lattice_ave' --output_path '../outputs/output_irqa_lattice_ave' \
--model_param '{"pooling_mode":"ave", "layer_num":2}'

python3 run_nlpcc.py -m model_nlpcc_kbre_cws \
--train_base_func deal_nlpcckbqa_mws_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_mws --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 8000 --train_epoch_num 40 --recover_func 'recover_kbqa' \
--model_path '../models/model_kbre_lattice_gated' --output_path '../outputs/output_kbre_lattice_gated' \
--model_param '{"pooling_mode":"gated", "layer_num":1}'

python3 run_nlpcc.py -m model_nlpcc_irqa_cws \
--train_base_func deal_nlpccirqa_mws_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_mws --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' \
--train_data_size 3000 --train_epoch_num 25 --recover_func 'recover_dbqa' \
--model_path '../models/model_irqa_lattice_gated' --output_path '../outputs/output_irqa_lattice_gated' \
--model_param '{"pooling_mode":"gated", "layer_num":2}'







