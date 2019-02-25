#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import os
import json
import time
import codecs
import random
import argparse

import numpy as np

from tqdm import tqdm
from importlib import import_module

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)

# 设置session
KTF.set_session(sess)


global args
def getArgs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--model', '-m',     type=str, required=True, help='model file name in .model/')
    parse.add_argument('--model_param',     type=str, default=None, help='model parameters, in form of str(json)')
    parse.add_argument('--w2v_path',        type=str, help='word2vec file path, google w2v form', default='../data/vectorsw300l20.all')
    parse.add_argument('--train',           type=str, required=True, help='train data path')
    parse.add_argument('--test',            type=str, required=True, help='test data path')
    
    parse.add_argument('--model_path',      type=str, required=True, help='model files path')
    parse.add_argument('--output_path',      type=str, required=True, help='output files path')
    parse.add_argument('--log_path',        type=str, default=None, help='model files path')
    
    parse.add_argument('--train_base_func', type=str, default=None, help='train data basic deal function name, a member of CPreprocess, default is lambda x : x')
    parse.add_argument('--train_deal_func', type=str, default=None, help='train data deal function name, a member of CPreprocess, default is lambda x : x')
    parse.add_argument('--train_pad_func',  type=str, default=None, help='train data pad function name, a member of CPreprocess, default is lambda x : x')
    parse.add_argument('--test_base_func',  type=str, default=None, help='test form, default is the same as train form, basic deal, usually same as train')
    parse.add_argument('--test_deal_func',  type=str, default=None, help='test form, default is the same as train form')
    parse.add_argument('--test_pad_func',   type=str, default=None, help='test form, default is the same as train form')
    
    parse.add_argument('--need_train',      type=str, default='True', help='need to train')
    parse.add_argument('--need_test',       type=str, default='True', help='need to test')

    parse.add_argument('--recover_func',    type=str, default='recover_kbqa', help='recover function, combine NN results to raw datas, for evaluate and output.')
    parse.add_argument('--gpu_fraction',    type=int, default=1, help='n=gpu_fraction, then the memory will be 1/n of all. dafault=1')
    parse.add_argument('--train_data_size', type=int, default=8000, help='train data size')
    parse.add_argument('--train_batch_size', type=int, default=64, help='train batch size')
    parse.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parse.add_argument('--train_epoch_num', type=int, default=25, help='train epoch num')
    parse.add_argument('--start_epoch_num', type=int, default=-1, help='train epoch num')

    args_ret=parse.parse_args()
    print(args_ret)

    return vars(args_ret)
if __name__ == '__main__':
    args = getArgs()

    """   gpu_fraction   """
    if not args['gpu_fraction'] == 1 :
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf

        rate_t = .9 / args['gpu_fraction']
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = rate_t
        # config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

    """   other parameters   """
    args['train_batch_size'] = args['train_batch_size'] # 64
    args['train_data_size'] = args['train_data_size'] # 8000
    args['test_batch_size'] = args['test_batch_size'] # 64
    args['train_epoch_num'] = args['train_epoch_num'] # 25 
    args['need_train'] = args['need_train'].lower() in ['true', 't', 'truth']
    args['need_test'] = args['need_test'].lower() in ['true', 't', 'truth']

    """   model and log files path   """
    if not os.path.exists(args['model_path']) :
        os.makedirs(args['model_path'])
    if args['log_path'] is None :
        for i in range(10000) :
            log_name = os.path.join(args['model_path'], 'log_%d'%(i))
            if os.path.exists(log_name) :
                continue
            break
        args['log_path'] = log_name
    args['log'] = codecs.open(args['log_path'], 'w', 'utf-8')

    """   output files path   """
    if not os.path.exists(args['output_path']) :
        os.makedirs(args['output_path'])

    global get_model, preprocess
    """   import get_model function   """
    sys.path.append('model')
    model_file = import_module(args['model'])
    if args['model_param'] is None :
        get_model = model_file.get_model
    else :
        args['model_param'] = json.loads(args['model_param'])
        get_model = lambda x : model_file.get_model(x, args['model_param'])

    """   import recover function   """
    import recover_datas
    recover = eval('recover_datas.%s'%(args['recover_func']))

    """   deal preprocess object   """
    from preprocess import CPreprocess
    from data_delegate import CData

    preprocess = CPreprocess(w2v_path = args['w2v_path'])
    args['train_base_func'] = getattr(preprocess, args['train_base_func']) if not args['train_base_func'] is None else lambda x : x
    args['train_deal_func'] = getattr(preprocess, args['train_deal_func']) if not args['train_deal_func'] is None else lambda x : x
    args['train_pad_func']  = getattr(preprocess, args['train_pad_func']) if not args['train_pad_func'] is None else lambda x : x
    args['test_base_func']  = getattr(preprocess, args['test_base_func']) if not args['test_base_func'] is None else args['train_base_func']
    args['test_deal_func']  = getattr(preprocess, args['test_deal_func']) if not args['test_deal_func'] is None else args['train_deal_func']
    args['test_pad_func']   = getattr(preprocess, args['test_pad_func']) if not args['test_pad_func'] is None else args['train_pad_func']

    print(args)
        
if __name__ == '__main__':
    """   定义数据   """
    data_train = CData(args['train'], default_size=args['train_data_size'])
    data_test  = CData(args['test'])
    to_tests = {
        #'data_train' : data_train, 
        'data_test'  : data_test, 
    }

    """   定义模型和数据预处理模式   """
    model = get_model(preprocess.get_w2v_matrix())
    get_train_data = lambda x : x.get_data_generator(
        args['train_pad_func'], args['train_batch_size'], 
        deal_base_func=args['train_base_func'], 
        deal_function=args['train_deal_func'], 
        verbose=True)
    get_test_data  = lambda x : x.get_data_generator(
        args['test_pad_func'], args['test_batch_size'], 
        deal_base_func=args['test_base_func'], 
        deal_function=args['test_deal_func'], 
        size=False, shuffle=False, verbose=True)

    # """   测试数据默认不需要每轮更新~   """
    # for k in to_tests :
    #     to_tests[k] = list(get_test_data(to_tests[k])) + [to_tests[k]]

    train_loss = ' '
    test_loss = ' '
    """   epoches 循环   """
    for iters in range(args['train_epoch_num']) :
        if iters < args['start_epoch_num'] :
            continue
        print('epoches : %d'%(iters))
        model_path_t = os.path.join(args['model_path'], 'model_%d'%(iters))

        """   train   """
        if args['need_train'] :
            train_generator = get_train_data(data_train)
            history = model.fit_generator(train_generator, data_train.it_size, epochs=1, shuffle=False)
            train_loss = str(history.history['loss'])
            model.save_weights(model_path_t)
        else :
            model.load_weights(model_path_t)

        """   test & output   """
        if args['need_test'] :
            test_loss = {}
            kv_list = [[k, to_tests[k]] for k in to_tests]
            for name, data in kv_list :
                test_generator = get_test_data(data)

                # simple_evaluate
                # test_results = model.evaluate_generator(test_generator, data.it_size, verbose=1)

                # evaluate & log
                test_predicted = model.predict_generator(test_generator, data.it_size, verbose=1)
                data_recovered, scores = recover(data.get_data(), test_predicted.tolist())

                output_path = os.path.join(args['output_path'], '%s_%d.json'%(name, iters))
                json.dump(data_recovered, codecs.open(output_path, 'w', 'utf-8'))

                test_loss[name] = scores
            test_loss = str(test_loss)
        
        """   write log   """
        args['log'].write('epoches : %d'%(iters)+'\n')
        args['log'].write(train_loss+'\n')
        args['log'].write(test_loss+'\n\n')
        args['log'].flush()
    args['log'].flush()