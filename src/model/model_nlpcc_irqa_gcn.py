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
from model_nlpcc_kbre_gcn import gcn_layer
get_shape = lambda x : str(x.shape)


def get_model(wordMatrix, model_param=None) :
    if model_param is None :
        model_param = {'layer_num':2}
    if not 'pooling_mode' in model_param :
        model_param['pooling_mode']='gcn'
    if not 'if_d' in model_param :
        model_param['if_d'] = True
    if not 'n_param' in model_param :
        model_param['n_param'] = 1024
    qu_l  = 80
    pre_l = 160
    pad_l = 5

    """   inputs   """
    qu_ids    = Input(shape=(qu_l,), dtype='int32', name='qu')
    pre_ids   = Input(shape=(pre_l,), dtype='int32', name='sent')
    qu_feature   = Input(shape=(qu_l,), dtype='float32', name='qu_feature')
    pre_feature = Input(shape=(pre_l,), dtype='float32', name='sent_feature')
    input_layers = [qu_ids, pre_ids, qu_feature, pre_feature]
    # (?, 50) (?, 20)

    """   embedding   """
    input_dim, output_dim = np.shape(wordMatrix)
    embedding = Embedding(input_dim, output_dim, weights=[wordMatrix], name='embedding')

    qu_words = embedding(qu_ids)
    pre_words = embedding(pre_ids)
    # (?, 50, 300) (?, 20, 300)

    qu_feature_c = Reshape((qu_l, 1))(qu_feature)
    pre_feature_c = Reshape((pre_l, 1))(pre_feature)
    qu_words  = Concatenate()([qu_words, qu_feature_c])
    pre_words = Concatenate()([pre_words, pre_feature_c])
    # (?, 60, 301) (?, 120, 301)

    """   index_inputs   """
    qu_mid_r   = Input(shape=(qu_l,), dtype='int32', name='qu_mid')
    qu_head  = Input(shape=(qu_l,pad_l), dtype='int32', name='qu_head')
    qu_tail  = Input(shape=(qu_l,pad_l), dtype='int32', name='qu_tail')
    pre_mid_r  = Input(shape=(pre_l,), dtype='int32', name='sent_mid')
    pre_head = Input(shape=(pre_l,pad_l), dtype='int32', name='sent_head')
    pre_tail = Input(shape=(pre_l,pad_l), dtype='int32', name='sent_tail')
    qu_i_ids = [qu_head, qu_mid_r, qu_tail]
    pre_i_ids = [pre_head, pre_mid_r, pre_tail]
    input_layers += [qu_head, qu_mid_r, qu_tail, pre_head, pre_mid_r, pre_tail]

    expand_dim = Lambda(lambda x:K.expand_dims(x, -1), output_shape=lambda x:x+(1,))
    qu_i_ids[1] = expand_dim(qu_i_ids[1])
    pre_i_ids[1] = expand_dim(pre_i_ids[1])

    """   convolution   """              
    n_param = model_param['n_param']#1024                                                                                                                       
    cnn1_head = Dense(n_param, name='cnn1_head')
    cnn1_mid  = Dense(n_param, name='cnn1_mid')  if model_param['if_d'] else cnn1_head
    cnn1_tail = Dense(n_param, name='cnn1_tail') if model_param['if_d'] else cnn1_head
    cnn1 = [cnn1_head, cnn1_mid, cnn1_tail]
    gate_dense=None
    if model_param['pooling_mode'] == 'gat' :
        base_gate = Dense(1, name='gate1_head', activation='sigmoid')
        gate_dense = [
            base_gate,
            Dense(1, name='gate1_mid', activation='sigmoid')  if model_param['if_d'] else base_gate,
            Dense(1, name='gate1_tail', activation='sigmoid') if model_param['if_d'] else base_gate,
        ]
    elif model_param['pooling_mode'] in ['gat_n', 'gat_2'] :
        base_gate = Dense(1, name='gate1_head', activation='sigmoid')
        gate_dense = [
            base_gate,
            Dense(1, name='gate1_mid', activation=None)  if model_param['if_d'] else base_gate,
            Dense(1, name='gate1_tail', activation=None) if model_param['if_d'] else base_gate,
        ]

    qu_results = gcn_layer(qu_i_ids, qu_words, cnn1, mode=model_param['pooling_mode'], gate_dense=gate_dense)
    pre_results = gcn_layer(pre_i_ids, pre_words, cnn1, mode=model_param['pooling_mode'], gate_dense=gate_dense)
    # (?, 50, 1024)
    # (?, 20, 1024)

    for i in range(model_param['layer_num']-1) :
        cnn2_head = Dense(n_param, name='cnn%d_head'%(i+2))
        cnn2_mid  = Dense(n_param, name='cnn%d_mid'%(i+2))  if model_param['if_d'] else cnn2_head
        cnn2_tail = Dense(n_param, name='cnn%d_tail'%(i+2)) if model_param['if_d'] else cnn2_head
        cnn2 = [cnn2_head, cnn2_mid, cnn2_tail]
        gate_dense_2=None
        if model_param['pooling_mode'] == 'gat' :                                                                                                               
            base_gate_2 = Dense(1, name='gate%d_head'%(i+2), activation='sigmoid')
            gate_dense_2 = [
                base_gate_2,
                Dense(1, name='gate%d_mid'%(i+2), activation='sigmoid')  if model_param['if_d'] else base_gate_2,
                Dense(1, name='gate%d_tail'%(i+2), activation='sigmoid') if model_param['if_d'] else base_gate_2,
            ]
        elif model_param['pooling_mode'] in ['gat_n', 'gat_2'] :
            base_gate_2 = Dense(1, name='gate%d_head'%(i+2), activation='sigmoid')
            gate_dense_2 = [
                base_gate_2,
                Dense(1, name='gate%d_mid'%(i+2), activation=None)  if model_param['if_d'] else base_gate_2,
                Dense(1, name='gate%d_tail'%(i+2), activation=None) if model_param['if_d'] else base_gate_2,
            ]


        qu_results_2 = gcn_layer(qu_i_ids, qu_results, cnn2,
            active=None, mode = model_param['pooling_mode'], gate_dense=gate_dense_2)
        pre_results_2 = gcn_layer(pre_i_ids, pre_results, cnn2,
            active=None, mode = model_param['pooling_mode'], gate_dense=gate_dense_2)
        # (?, 80, 1024)
        # (?, 160, 1024)

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
    """?~M~U?~E~C?~K?~U"""

    # sys.path.append('..')
    # from basic import *
    # w2v = W2V('../../data/vectorsw300l20.all')
    # wordMatrix = w2v.getMatrix()
    # # (155841, 300)

    wordMatrix = np.random.random([155841, 300])

    model = get_model(wordMatrix)

    model.summary()




