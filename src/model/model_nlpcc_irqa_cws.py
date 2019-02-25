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

import keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Embedding, Activation, Dense, Flatten, Dropout, Lambda, Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Concatenate, Multiply, Add
import keras.backend as K

from model_base import *
from model_nlpcc_kbre_cws import get_ig_inputs, one_layer
get_shape = lambda x : str(x.shape)

def get_para() :
    paras = {
        'qu_l' : 80,
        'sent_l' : 160,
    }
    paras['qu_i'] = [
        [[1,paras['qu_l']]], 
        [[1,paras['qu_l']], [2,100], [3,60], [6,60], [12,60]], 
        [[1,paras['qu_l']], [2,120], [3,90], [6,180], [12,180], [30,150]], 
    ]
    paras['sent_i'] = [
        [[1,paras['sent_l']]], 
        [[1,paras['sent_l']], [2,200], [3,180], [6,120], [12,60]], 
        [[1,paras['sent_l']], [2,240], [3,180], [6,360], [12,360], [30,150]],
    ]
    paras['qu_sum1'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][0]])
    paras['qu_sum2'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][1]])
    paras['qu_sum3'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][2]])
    paras['sent_sum1'] = sum([int(t[1]/t[0]) for t in paras['sent_i'][0]])
    paras['sent_sum2'] = sum([int(t[1]/t[0]) for t in paras['sent_i'][1]])
    paras['sent_sum3'] = sum([int(t[1]/t[0]) for t in paras['sent_i'][2]])
    """
    # print (json.dumps(paras, indent=2))
    "qu_sum1": 80,
    "qu_sum2": 165,
    "qu_sum3": 220,
    "sent_sum1": 160,
    "sent_sum2": 345,
    "sent_sum3": 435
    """
    # print(paras)
    return paras

def get_model(wordMatrix, model_param=None) :
    if model_param is None :
        model_param = {'layer_num':2}
    if not 'pooling_mode' in model_param :
        model_param['pooling_mode'] = 'max'
    paras = get_para()

    """   inputs   """
    qu_ids    = Input(shape=(paras['qu_l'],), dtype='int32', name='qu')
    pre_ids   = Input(shape=(paras['sent_l'],), dtype='int32', name='sent')
    qu_feature   = Input(shape=(paras['qu_l'],), dtype='float32', name='qu_feature')
    pre_feature = Input(shape=(paras['sent_l'],), dtype='float32', name='sent_feature')
    input_layers = [qu_ids, pre_ids, qu_feature, pre_feature]
    # (?, 50) (?, 20)

    """   embedding   """
    input_dim, output_dim = np.shape(wordMatrix)
    embedding = Embedding(input_dim, output_dim, weights=[wordMatrix], name='embedding')

    qu_words = embedding(qu_ids)
    pre_words = embedding(pre_ids)
    # (?, 50, 300) (?, 20, 300)

    qu_feature_c = Reshape((paras['qu_l'], 1))(qu_feature)
    pre_feature_c = Reshape((paras['sent_l'], 1))(pre_feature)
    qu_words  = Concatenate()([qu_words, qu_feature_c])
    pre_words = Concatenate()([pre_words, pre_feature_c])
    # (?, 60, 301) (?, 120, 301)

    """   index_inputs   """
    qu_convert1 = Input(shape=(paras['qu_sum1'],paras['qu_l']), dtype='float32', name='qu_convert1')
    qu_convert2 = Input(shape=(paras['qu_sum2'],paras['qu_l']), dtype='float32', name='qu_convert2')
    qu_convert3 = Input(shape=(paras['qu_sum3'],paras['qu_l']), dtype='float32', name='qu_convert3')
    pre_convert1 = Input(shape=(paras['sent_sum1'],paras['sent_l']), dtype='float32', name='sent_convert1')
    pre_convert2 = Input(shape=(paras['sent_sum2'],paras['sent_l']), dtype='float32', name='sent_convert2')
    pre_convert3 = Input(shape=(paras['sent_sum3'],paras['sent_l']), dtype='float32', name='sent_convert3')
    input_layers += [qu_convert1, qu_convert2, qu_convert3, pre_convert1, pre_convert2, pre_convert3] # 这里写成float只是算梯度什么的比较方便~

    qu_uig_group  = get_ig_inputs(1, paras['qu_i'][0], 'qu')
    qu_big_group  = get_ig_inputs(2, paras['qu_i'][1], 'qu')
    qu_tig_group  = get_ig_inputs(3, paras['qu_i'][2], 'qu')
    pre_uig_group = get_ig_inputs(1, paras['sent_i'][0], 'sent')
    pre_big_group = get_ig_inputs(2, paras['sent_i'][1], 'sent')
    pre_tig_group = get_ig_inputs(3, paras['sent_i'][2], 'sent')
    qu_igs  = [qu_uig_group, qu_big_group, qu_tig_group]
    pre_igs = [pre_uig_group, pre_big_group, pre_tig_group]

    input_layers += qu_uig_group+qu_big_group+qu_tig_group+pre_uig_group+pre_big_group+pre_tig_group
    # print(len(input_layers))
    # for t in input_layers :
    #     print(t.name, get_shape(t))
    """
    22
    qu:0 (?, 50)    pre:0 (?, 20)

    qu_convert1:0 (?, 50, 50)   qu_convert2:0 (?, 76, 50)   qu_convert3:0 (?, 92, 50)
    pre_convert1:0 (?, 20, 20)  pre_convert2:0 (?, 46, 20)  pre_convert3:0 (?, 62, 20)
    
    qu_uig_1:0 (?, 50, 1)
    qu_big_1:0 (?, 50, 2)   qu_big_3:0 (?, 60, 2)   qu_big_10:0 (?, 60, 2)
    qu_tig_1:0 (?, 50, 3)   qu_tig_3:0 (?, 90, 3)   qu_tig_10:0 (?, 90, 3)  qu_tig_30:0 (?, 90, 3)

    pre_uig_1:0 (?, 20, 1)  
    pre_big_1:0 (?, 20, 2)  pre_big_3:0 (?, 60, 2)  pre_big_10:0 (?, 60, 2)
    pre_tig_1:0 (?, 20, 3)  pre_tig_3:0 (?, 90, 3)  pre_tig_10:0 (?, 90, 3) pre_tig_30:0 (?, 90, 3)
    """

    """   convolution   """
    cnn1_0 = Dense(256, name='cnn1_0')
    cnn1_1 = Dense(512, name='cnn1_1')
    cnn1_2 = Dense(256, name='cnn1_2')
    cnn1 = [cnn1_0, cnn1_1, cnn1_2]
    if model_param['pooling_mode'] == 'attentive' :
        attentive_denses = [Dense(t, use_bias=False, name='a_dense_1_%d'%(i)) 
            for i, t in enumerate([256, 512, 256])]
    elif model_param['pooling_mode'] == 'gated' :
        attentive_denses = [Dense(t, activation=None, name='a_dense_1_%d'%(i)) 
            for i, t in enumerate([1, 1, 1])]
    else :
        attentive_denses = None


    qu_results = one_layer(qu_igs, qu_words, cnn1, paras['qu_i'], converts=[qu_convert1, qu_convert2, qu_convert3], 
        pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses)
    pre_results = one_layer(pre_igs, pre_words, cnn1, paras['sent_i'], converts=[pre_convert1, pre_convert2, pre_convert3], 
        pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses)
    # (?, 50, 1024)
    # (?, 20, 1024)
    
    for i in range(model_param['layer_num']-1) :
        cnn2_0 = Dense(256, name='cnn%d_0'%(i+2))
        cnn2_1 = Dense(512, name='cnn%d_1'%(i+2))
        cnn2_2 = Dense(256, name='cnn%d_2'%(i+2))
        cnn2 = [cnn2_0, cnn2_1, cnn2_2]
        if model_param['pooling_mode'] == 'attentive' :
            attentive_denses_t = [Dense(t, use_bias=False, name='a_dense_%d_%d'%(i+2, j)) 
                for j, t in enumerate([256, 512, 256])]
        elif model_param['pooling_mode'] == 'gated' :
            attentive_denses_t = [Dense(t, activation=None, name='a_dense_%d_%d'%(i+2, j)) 
                for j, t in enumerate([1, 1, 1])]
        else :
            attentive_denses_t = None


        qu_results_2 = one_layer(qu_igs, qu_results, cnn2, paras['qu_i'], converts=[qu_convert1, qu_convert2, qu_convert3],
            active=None, pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses_t)
        pre_results_2 = one_layer(pre_igs, pre_results, cnn2, paras['sent_i'], converts=[pre_convert1, pre_convert2, pre_convert3],
            active=None, pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses_t)
        # (?, 50, 1024)
        # (?, 20, 1024)

        qu_results_2 = Activation('relu')(Add()([qu_results, qu_results_2]))
        pre_results_2 = Activation('relu')(Add()([pre_results, pre_results_2]))
        qu_results = qu_results_2
        pre_results = pre_results_2


    """   merge   """
    label = count_similarity(qu_results, pre_results)
    # (?, 1)

    model = Model(inputs=input_layers, outputs=[label])
    # from keras.utils import multi_gpu_model
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss={'label':'binary_crossentropy'}, optimizer='adadelta') #my_loss_llh adadelta
    
    return model

if __name__ == '__main__':
    pass
    """单元测试"""

    # sys.path.append('..')
    # from basic import *
    # w2v = W2V('../../data/vectorsw300l20.all')
    # wordMatrix = w2v.getMatrix()
    # # (155841, 300)

    wordMatrix = np.random.random([155841, 300])

    model = get_model(wordMatrix)

    model.summary()






