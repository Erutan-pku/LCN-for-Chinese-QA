###  download pretrained models
mkdir -p models/best_irqa_model
mkdir -p models/best_kbre_model

wget http://59.108.48.35/aaai19_lattice/best_irqa_model/model_13 -O models/best_irqa_model/model_13
wget http://59.108.48.35/aaai19_lattice/best_kbre_model/model_32 -O models/best_kbre_model/model_32

###  test :
cd src

# lattice max-pooling for kbre
python3 run_nlpcc.py -m model_nlpcc_kbre_cws \
--train_base_func deal_nlpcckbqa_mws_base --train_deal_func deal_nlpcckbqa_train --train_pad_func padding_nlpcckbqa_mws --test_deal_func deal_nlpcckbqa_test \
--train '../data/train_re_14262_segmws.json' --test '../data/test_re_9413_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' --need_train 'False' \
--train_data_size 8000  --recover_func 'recover_kbqa' --start_epoch_num 32 --train_epoch_num 33 \
--model_path '../models/best_kbre_model' --output_path '../outputs/best_kbre_output' \
--model_param '{"pooling_mode":"max", "layer_num":1}'

# lattice gated for irqa
python3 run_nlpcc.py -m model_nlpcc_irqa_cws \
--train_base_func deal_nlpccirqa_mws_base --train_deal_func deal_nlpccirqa_train --train_pad_func padding_nlpccirqa_mws --test_deal_func deal_nlpccirqa_test \
--train '../data/train_dbqa_8768_segmws.json' --test '../data/test_dbqa_5997_segmws.json' \
--w2v_path '../data_vectors/vectorsw300l20.all' --need_train 'False' \
--train_data_size 3000 --recover_func 'recover_dbqa' --start_epoch_num 13 --train_epoch_num 14 \
--model_path '../models/best_irqa_model' --output_path '../outputs/best_irqa_output' \
--model_param '{"pooling_mode":"gated", "layer_num":2}'


########################################################################################
# notes: I did not find the original models, so I just find models that perform similarly. 
#        So minute gaps exists.
# 
# lcn+max for kbre :
# results : {'data_test': {'r@3': 0.9833209391267396, 'r@1': 0.9362583660894508, 'len': 9413, 'MRR': 0.9609203143096297, 'r@5': 0.9915011154785934, 'r@10': 0.9971316264740253, 'r@2': 0.972803569531499, 'len_pre': 155593}}
# compare with paper :
# paper :  93.54% .9604         this : 93.62% .9609
# 
# lcn+gate for irqa :
# results : {'data_test': {'MAP': 0.8894401104336473, 'r@5': 0.9603134900783725, 'r@2': 0.9114557278639319, 'len': 5997, 'len_sent': 122531, 'r@3': 0.9379689844922461, 'r@1': 0.832416208104052, 'MRR': 0.8901682462668358}}
# compare with paper :
# paper :  .8895 .8902 83.24%   this : .8894 .8902 83.24%
# 
########################################################################################










