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

get_shape = lambda x : str(x.shape)

def gcn_layer(indexs, words, dense_list, active='relu', mode='gcn', gate_dense=None) :
    assert mode in ['gcn', 'maxlcn', 'avelcn', 'gat_n', 'gcn_max']
    len_words = words.shape[1]
    # print(words) (?, 50, 300)

    """   embedding   """
    onehot_t = Lambda(lambda x : K.one_hot(x, len_words),
        output_shape=lambda input_shape: input_shape+(len_words,))
    emb_lookup  = Lambda(lambda x : tf.einsum('abci,aij->abcj', x[0], x[1]))
    index_emb = [emb_lookup([onehot_t(t), words]) for t in indexs]
    # print(', '.join(map(get_shape, index_emb)))
    # (?, 50, 5, 300), (?, 50, 1, 300), (?, 50, 5, 300)

    """   cnn   """
    index_cnn = []
    for i, t in enumerate(index_emb) :
        index_cnn.append(dense_list[i](t))
    # print(', '.join(map(get_shape, index_cnn)))
    # (?, 50, 5, 1024), (?, 50, 1, 1024), (?, 50, 5, 1024)

    """   combine   """
    def get_indicator(index_all) :
        reduce_index = tf.sign(tf.abs(index_all))
        reduce_index = tf.cast(reduce_index, 'float32')
        reduce_index = tf.expand_dims(reduce_index, -1)                                                                                                         
        return reduce_index

    def pooling_spt_ave(cnn_all, index_all) :
        # Tensor("concatenate_1/concat:0", shape=(?, 50, 11), dtype=int32)
        # Tensor("concatenate_2/concat:0", shape=(?, 50, 11, 1024), dtype=float32)

        indicator_all = get_indicator(index_all)
        # Tensor("lambda_4/ExpandDims:0", shape=(?, 50, 11, 1), dtype=float32)
        cnn_msk = cnn_all * indicator_all

        cnn_result    = tf.reduce_sum(cnn_msk, axis=-2)
        indicator_sum = tf.reduce_sum(indicator_all, axis=-2)
        indicator_sum = indicator_sum+tf.cast(tf.ones_like(indicator_sum), 'float32')/100000.
        # Tensor("lambda_4/Sum:0", shape=(?, 50, 1024), dtype=float32)
        # Tensor("lambda_4/Sum_1:0", shape=(?, 50, 1), dtype=float32)
        cnn_result    = cnn_result / indicator_sum

        return cnn_result

    def get_lattice_max(cnn_all) :
        maxed_cnn_all = [tf.reduce_max(t, axis=-2) for t in cnn_all]
        result = Add()(maxed_cnn_all)
        return result
    def get_lattice_ave(cnn_all, index_all) :                                                                                                                   
        # cnn_all : (?, 50, 5, 1024), (?, 50, 1, 1024), (?, 50, 5, 1024)

        indicator_all = list(map(get_indicator, index_all))
        # tf.Tensor 'lambda_4/ExpandDims:0' shape=(?, 50, 5, 1) dtype=float32
        # tf.Tensor 'lambda_4/ExpandDims_1:0' shape=(?, 50, 1, 1) dtype=float32
        # tf.Tensor 'lambda_4/ExpandDims_2:0' shape=(?, 50, 5, 1) dtype=float32
        indicator_all_sumed = [tf.reduce_max(t, axis=-2) for t in indicator_all]
        indicator_all_sumed = [t + tf.ones_like(t)/100000. for t in indicator_all_sumed]
        # Tensor("lambda_4/Max:0", shape=(?, 50, 1), dtype=float32)
        # Tensor("lambda_4/Max_1:0", shape=(?, 50, 1), dtype=float32)
        # Tensor("lambda_4/Max_2:0", shape=(?, 50, 1), dtype=float32)

        cnn_all_sumed = [tf.reduce_sum(t*indicator_all[i], axis=-2)/indicator_all_sumed[i] for i, t in enumerate(cnn_all)]
        # (?, 50, 1024), (?, 50, 1024), (?, 50, 1024)
        result = Add()(cnn_all_sumed)
        return result


    if mode == 'gcn' :
        index_all = Concatenate(axis=2)(indexs)
        cnn_all   = Concatenate(axis=2)(index_cnn)
        # Tensor("concatenate_1/concat:0", shape=(?, 50, 11), dtype=int32)
        # Tensor("concatenate_2/concat:0", shape=(?, 50, 11, 1024), dtype=float32)

        results = Lambda(lambda x:pooling_spt_ave(x, index_all))(cnn_all)
    elif mode == 'gcn_max' :
        cnn_all   = Concatenate(axis=2)(index_cnn)
        # Tensor("concatenate_1/concat:0", shape=(?, 50, 11, 1024), dtype=float32)

        results = Lambda(lambda x:tf.reduce_max(x, axis=-2))(cnn_all)
    elif mode == 'gat_n' :                                                                                                                                        
        # softmax version
        gate_all = []
        for i, t in enumerate(index_cnn) :
            gate_i = gate_dense[i](t)
            gate_all.append(gate_i)

        gate_all = Concatenate(axis=2)(gate_all)
        cnn_all   = Concatenate(axis=2)(index_cnn)
        # Tensor("concatenate_1/concat:0", shape=(?, 50, 11, 1), dtype=float32)
        # Tensor("concatenate_2/concat:0", shape=(?, 50, 11, 1024), dtype=float32)
        def get_weighted_sum(cnn_all, gate_all) :
            index_all = Concatenate(axis=2)(indexs)
            indicator_all = get_indicator(index_all)
            # Tensor("lambda_4/ExpandDims:0", shape=(?, 50, 11, 1), dtype=float32)

            gate_all = gate_all - (tf.ones_like(indicator_all)-indicator_all) * 1e6
            
            gate_all = tf.nn.softmax(gate_all, axis=-2)
            cnn_all = cnn_all * gate_all
            cnn_all = tf.reduce_sum(cnn_all, axis=-2)
            return cnn_all

        results = Lambda(lambda x:get_weighted_sum(x[0], x[1]))([cnn_all, gate_all])
    elif mode == 'maxlcn' :
        results = Lambda(get_lattice_max)(index_cnn)
    elif mode == 'avelcn' :
        results = Lambda(lambda x : get_lattice_ave(x, indexs))(index_cnn)
    return results

def get_model(wordMatrix, model_param=None) :
    if model_param is None :
        model_param = {'layer_num':1}
    if not 'pooling_mode' in model_param :
        model_param['pooling_mode']='gat_2'#'gcn'
    if not 'if_d' in model_param :
        model_param['if_d'] = True
    if not 'n_param' in model_param :
        model_param['n_param'] = 1024
    # print(json.dumps(model_param))                                                                                                                            
    # {"layer_num": 1, "pooling_mode": "gcn", "if_d": true}
    qu_l  = 50
    pre_l = 20
    pad_l = 5

    """   inputs   """
    qu_ids    = Input(shape=(qu_l,), dtype='int32', name='qu')
    pre_ids   = Input(shape=(pre_l,), dtype='int32', name='pre')
    input_layers = [qu_ids, pre_ids]
    # (?, 50) (?, 20)

    qu_mid_r   = Input(shape=(qu_l,), dtype='int32', name='qu_mid')
    qu_head  = Input(shape=(qu_l,pad_l), dtype='int32', name='qu_head')
    qu_tail  = Input(shape=(qu_l,pad_l), dtype='int32', name='qu_tail')
    pre_mid_r  = Input(shape=(pre_l,), dtype='int32', name='pre_mid')
    pre_head = Input(shape=(pre_l,pad_l), dtype='int32', name='pre_head')
    pre_tail = Input(shape=(pre_l,pad_l), dtype='int32', name='pre_tail')
    qu_i_ids = [qu_head, qu_mid_r, qu_tail]
    pre_i_ids = [pre_head, pre_mid_r, pre_tail]
    input_layers += [qu_head, qu_mid_r, qu_tail, pre_head, pre_mid_r, pre_tail]

    expand_dim = Lambda(lambda x:K.expand_dims(x, -1), output_shape=lambda x:x+(1,))
    qu_i_ids[1] = expand_dim(qu_i_ids[1])
    pre_i_ids[1] = expand_dim(pre_i_ids[1])
    # Tensor("lambda_1/ExpandDims:0", shape=(?, 50, 1), dtype=int32)                                                                                            
    # Tensor("lambda_1_1/ExpandDims:0", shape=(?, 20, 1), dtype=int32)

    """   embedding   """
    input_dim, output_dim = np.shape(wordMatrix)
    embedding = Embedding(input_dim, output_dim, weights=[wordMatrix], name='embedding')

    qu_words = embedding(qu_ids)
    pre_words = embedding(pre_ids)
    # (?, 50, 300) (?, 20, 300)

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
    # (?, 80, 1024)
    # (?, 160, 1024)

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
    model.compile(loss={'label':'binary_crossentropy'}, optimizer='adadelta')

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


