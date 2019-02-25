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

def get_para() :
    paras = {
        'qu_l' : 50,
        'pre_l' : 20,
    }
    paras['qu_i'] = [
        [[1,paras['qu_l']]], 
        [[1,paras['qu_l']], [3,60], [10,60]], 
        [[1,paras['qu_l']], [3,90], [10,90], [30,90]], 
    ]
    paras['pre_i'] = [
        [[1,paras['pre_l']]], 
        [[1,paras['pre_l']], [3,60], [10,60]], 
        [[1,paras['pre_l']], [3,90], [10,90], [30,90]]
    ]
    paras['qu_sum1'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][0]])
    paras['qu_sum2'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][1]])
    paras['qu_sum3'] = sum([int(t[1]/t[0]) for t in paras['qu_i'][2]])
    paras['pre_sum1'] = sum([int(t[1]/t[0]) for t in paras['pre_i'][0]])
    paras['pre_sum2'] = sum([int(t[1]/t[0]) for t in paras['pre_i'][1]])
    paras['pre_sum3'] = sum([int(t[1]/t[0]) for t in paras['pre_i'][2]])
    """
    # print (json.dumps(paras, indent=2))
    "qu_sum1": 50,
    "qu_sum2": 76,
    "qu_sum3": 92,
    "pre_sum1": 20,
    "pre_sum2": 46,
    "pre_sum3": 62
    """

    return paras

def get_ig_inputs(n, ndfs, head_name) :
    """   
    获得一个input_list，如：
        get_ig_inputs(2, paras['qu_i'][1], 'qu')
    意为qu的bigram inputs，相关参数见 paras['qu_i'][1] = [[1,50], [3,60], [10,60]]

    返回一个Input list, len = len(ndfs)
    """
    n_dicts = {1:'u', 2:'b', 3:'t'}
    n_gram = n_dicts[n]
    to_ret = []

    for i, t in enumerate(ndfs) :
        name_t = '%s_%sig_%d'%(head_name, n_gram, t[0])
        input_t = Input(shape=(t[1], n), dtype='int32', name=name_t)
        to_ret.append(input_t)

    return to_ret

def get_conv_join(index, words, dense_layer, ndfs, active='relu', pooling_mode='max', attentive_dense=None) :
    """   
    对一个index_list做CNN，其中：
        index： 指示，       (?, 50, 3), (?, 90, 3), (?, 90, 3), (?, 90, 3)
        words： 词向量，     (?, 50, 300)
        dense_layer：CNN参数~ 如256个filter
        ndfs： index的原始参数，如 [[1,50], [3,90], [10,90], [30,90]], 用于max_pooling~

    返回一个tensor， 形如：
    """
    assert pooling_mode in ['max', 'ave', 'attentive', 'gated']

    len_words = words.shape[1]
    split_ns = [int(t.shape[1]) for t in index]
    # print(', '.join(map(get_shape, index)))
    # print(split_ns)
    # print(len_words, words.shape)
    # (?, 50, 3), (?, 90, 3), (?, 90, 3), (?, 90, 3)
    # [50, 90, 90, 90]
    # 50 (?, 50, 300)

    index_all = index[0] if len(index) == 1 else Concatenate(axis=1)(index)
    # (?, 320, 3)

    def get_indicator(index_all) :
        # index_all # (?, 320, 3)
        reduce_index = tf.reduce_sum(index_all, axis=-1, keepdims=True)
        # (?, 320, 1)
        reduce_index = tf.sign(tf.abs(reduce_index))
        reduce_index = tf.cast(reduce_index, 'float32')
        # (?, 320, 1)
        return reduce_index
    indicator_all = get_indicator(index_all)
    # this is a tf tensor

    """   embedding   """
    onehot_t = Lambda(lambda x : K.one_hot(x, len_words), 
        output_shape=lambda input_shape: tuple(list(input_shape)+[len_words]))
    emb_lookup  = Lambda(lambda x : tf.einsum('abci,aij->abcj', x[0], x[1]))
    index_emb = emb_lookup([onehot_t(index_all), words])
    # print(index_emb.shape)
    # (?, 320, 3, 300)

    """   cnn   """
    reshape = lambda x : Reshape([int(t) for t in list(x.shape[1:-2])+[x.shape[-2]*x.shape[-1]] ])(x)
    index_cnn = dense_layer(reshape(index_emb))
    # print(index_cnn.shape)
    # (?, 320, 256)

    index_cnn = Lambda(lambda x : indicator_all*x)(index_cnn)

    def pooling_spt(index_cnn, _=None) :
        index_cnn_list = tf.split(index_cnn, split_ns, axis=1)
        # print(', '.join(map(get_shape, index_cnns)))
        # (?, 50, 256), (?, 90, 256), (?, 90, 256), (?, 90, 256)
        index_pool = [MaxPooling1D(pool_size=ndfs[i][0])(t) for i, t in enumerate(index_cnn_list)]
        # print(', '.join(map(get_shape, index_pool)))
        # (?, 50, 256), (?, 30, 256), (?, 9, 256), (?, 3, 256)

        if len(index_pool) > 1:
            results = Concatenate(axis=1)(index_pool)
        else :
            results = index_pool[0]
        # print(results.shape)
        # (?, 92, 256)

        return results
    
    def pooling_spt_ave(index_cnn, indicator_all) :
        index_cnn_list = tf.split(index_cnn, split_ns, axis=1)
        indicator_list = tf.split(indicator_all, split_ns, axis=1)
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, indicator_list)))
        # (?, 50, 256), (?, 90, 256), (?, 90, 256), (?, 90, 256)
        # (?, 50, 1), (?, 90, 1), (?, 90, 1), (?, 90, 1)
        index_cnn_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(index_cnn_list)]
        indicator_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(indicator_list)]
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, indicator_list)))
        # (?, 50, 1, 256), (?, 30, 3, 256), (?, 9, 10, 256), (?, 3, 30, 256)
        # (?, 50, 1, 1), (?, 30, 3, 1), (?, 9, 10, 1), (?, 3, 30, 1)
        

        index_cnn_sum_list = [tf.reduce_sum(t, axis=-2) for t in index_cnn_list]
        indicator_sum_list = [tf.reduce_sum(t, axis=-2) for t in indicator_list]
        indicator_sum_list = [t+tf.cast(tf.ones_like(t), 'float32')/100000. for t in indicator_sum_list]
        # print(', '.join(map(get_shape, index_cnn_sum_list)))
        # print(', '.join(map(get_shape, indicator_sum_list)))
        # (?, 50, 256), (?, 30, 256), (?, 9, 256), (?, 3, 256)
        # (?, 50, 1), (?, 30, 1), (?, 9, 1), (?, 3, 1)

        index_pool = [t/indicator_sum_list[i] for i, t in enumerate(index_cnn_sum_list)]

        # index_pool = [MaxPooling1D(pool_size=ndfs[i][0])(t) for i, t in enumerate(index_cnn_list)]
        # print(', '.join(map(get_shape, index_pool)))
        # (?, 50, 256), (?, 30, 256), (?, 9, 256), (?, 3, 256)

        if len(index_pool) > 1:
            results = Concatenate(axis=1)(index_pool)
        else :
            results = index_pool[0]
        # print(results.shape)
        # (?, 92, 256)

        return results

    def pooling_spt_gated(index_cnn, attention_gate=None, indicator_all=None) :
        # index_cnn
        # (?, 50, 256)
        # # index_cnn = index_cnn * attention_gate
        # # (?, 50, 256)
        index_cnn_list = tf.split(index_cnn, split_ns, axis=1)
        attention_gate = attention_gate - (tf.ones_like(indicator_all)-indicator_all) * 1e6
        attention_gate_list = tf.split(attention_gate, split_ns, axis=1)
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, attention_gate_list)))
        # (?, 50, 256), (?, 90, 256), (?, 90, 256), (?, 90, 256)
        # (?, 50, 1), (?, 90, 1), (?, 90, 1), (?, 90, 1)

        index_cnn_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(index_cnn_list)]
        attention_gate_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(attention_gate_list)]
        attention_gate_list = [tf.nn.softmax(t, axis=-2) for t in attention_gate_list]
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, attention_gate_list)))
        # (?, 50, 1, 256), (?, 30, 3, 256), (?, 9, 10, 256), (?, 3, 30, 256)
        # (?, 50, 1, 1), (?, 30, 3, 1), (?, 9, 10, 1), (?, 3, 30, 1)
        index_cnn_list = [t*attention_gate_list[i] for i, t in enumerate(index_cnn_list)]

        index_cnn_sum_list = [tf.reduce_sum(t, axis=-2) for t in index_cnn_list]
        index_pool = index_cnn_sum_list
        # print(', '.join(map(get_shape, index_cnn_sum_list)))
        # (?, 50, 256), (?, 30, 256), (?, 9, 256), (?, 3, 256)

        if len(index_pool) > 1:
            results = Concatenate(axis=1)(index_pool)
        else :
            results = index_pool[0]
        # print(results.shape)
        # (?, 92, 256)

        return results

    def pooling_spt_attentive(index_cnn, w_index_cnn, indicator_all, mode_tt='max') :
        assert mode_tt in ['ave', 'max']
        index_cnn_list   = tf.split(index_cnn, split_ns, axis=1)
        w_index_cnn_list = tf.split(w_index_cnn, split_ns, axis=1)
        indicator_list   = tf.split(indicator_all, split_ns, axis=1)
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, w_index_cnn_list)))
        # print(', '.join(map(get_shape, indicator_list)))
        # (?, 50, 256), (?, 90, 256), (?, 90, 256), (?, 90, 256)
        # (?, 50, 256), (?, 90, 256), (?, 90, 256), (?, 90, 256)
        # (?, 50, 1), (?, 90, 1), (?, 90, 1), (?, 90, 1)

        index_cnn_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(index_cnn_list)]
        indicator_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(indicator_list)]
        w_index_cnn_list = [Reshape([int(int(t.shape[-2])/ndfs[i][0]), ndfs[i][0],t.shape[-1]])(t) for i, t in enumerate(w_index_cnn_list)]
        # print(', '.join(map(get_shape, index_cnn_list)))
        # print(', '.join(map(get_shape, w_index_cnn_list)))
        # print(', '.join(map(get_shape, indicator_list)))
        # (?, 50, 1, 256), (?, 30, 3, 256), (?, 9, 10, 256), (?, 3, 30, 256)
        # (?, 50, 1, 256), (?, 30, 3, 256), (?, 9, 10, 256), (?, 3, 30, 256)
        # (?, 50, 1, 1), (?, 30, 3, 1), (?, 9, 10, 1), (?, 3, 30, 1)

        index_pool = []
        for i, t in enumerate(w_index_cnn_list) :
            cnn_i       = index_cnn_list[i]
            weight_i    = t
            indicator_i = indicator_list[i] # (?, 3, 30, 1)


            weight_i = K.batch_dot(cnn_i, weight_i, axes=[3, 3])
            # (?, 3, 30, 30)
            """按位乘，先让超出部分的分布变成均匀分布"""
            # (?, 3, 30, 30) * (?, 3, 30, 1)
            weight_i = weight_i*indicator_i

            """再对每一个分布中不想要的点，减法抹去"""
            indicator_i_t = (K.ones_like(indicator_i) - indicator_i) * 1.e8
            ii_shape = list(map(int, indicator_i_t.shape[1:]))
            indicator_i_t = Reshape([ii_shape[0], ii_shape[2], ii_shape[1]])(indicator_i_t)
            # (?, 3, 1, 30)
            weight_i = K.softmax(weight_i-indicator_i_t)
            # (?, 3, 30, 30)

            """   加权求和，每个数的量纲还是一致的，空的数求和时没考虑，且对应的向量为真·均值   """
            # (?, 3, 30, 30), (?, 3, 30, 256)
            cnn_i_t = K.batch_dot(weight_i, cnn_i, axes=[3, 2])
            # (?, 3, 30, 256)

            """   再对self_attention的结果再做均值   """
            if mode_tt == 'ave' :
                cnn_i_t = tf.reduce_sum(cnn_i_t * indicator_i, axis=-2)
                indicator_sum_i = tf.reduce_sum(indicator_i, axis=-2)
                # (?, 3, 256), (?, 3, 1)

                indicator_sum_i = indicator_sum_i+tf.cast(tf.ones_like(indicator_sum_i), 'float32')/1.e8
                cnn_i_t = cnn_i_t / indicator_sum_i
            else :
                cnn_i_t = tf.reduce_max(cnn_i_t * indicator_i, axis=-2)
            index_pool.append(cnn_i_t)
        
        # print(', '.join(map(get_shape, index_pool)))
        # (?, 50, 256), (?, 30, 256), (?, 9, 256), (?, 3, 256)

        if len(index_pool) > 1:
            results = Concatenate(axis=1)(index_pool)
        else :
            results = index_pool[0]
        # print(results.shape)
        # (?, 92, 256)

        return results
    

    """   max_pooling   """
    # pooling_mode in ['max', 'ave', 'attentive']
    if pooling_mode == 'max' :
        results = Lambda(lambda x:pooling_spt(x))(index_cnn)
    elif pooling_mode == 'ave' :
        results = Lambda(lambda x:pooling_spt_ave(x, indicator_all))(index_cnn)
    elif pooling_mode == 'attentive' :
        dense_wt = attentive_dense
        w_index_cnn = dense_wt(index_cnn)
        # 上面那个是做双线性变换时用的参数。
        results = Lambda(lambda x:pooling_spt_attentive(x[0], x[1], indicator_all))([index_cnn, w_index_cnn])
    elif pooling_mode == 'gated' :
        attention_gate = attentive_dense(index_cnn)
        results = Lambda(lambda x:pooling_spt_gated(x[0], x[1], indicator_all))([index_cnn, attention_gate])

    # print(results.shape)
    # (?, 92, 256)
    if not active is None :
        results = Activation(active)(results)

    return results

def one_layer(indexs, words, dense_list, ndfs, converts, mode='join', active='relu', pooling_mode='max', attentive_dense=None) :
    """
    这个join并没有明显比single要快，估计single的并行性已经差不多了吧~
    """
    assert mode == 'join' 
    get_conv = get_conv_join
    
    assert len(dense_list) == len(ndfs) == len(converts)
    n_cnns = len(dense_list)

    if attentive_dense is None :
        attentive_dense = [None] * n_cnns

    temp_results = [get_conv(indexs[i], words, dense_list[i], ndfs[i], 
        active=active, pooling_mode=pooling_mode, attentive_dense=attentive_dense[i]) for i in range(3)]
    # print(', '.join(map(get_shape, temp_results)))
    # (?, 20, 256), (?, 46, 512), (?, 62, 256)

    back_lookup  = Lambda(lambda x : tf.einsum('aij,aib->ajb', x[0], x[1]))
    temp_results = [
      back_lookup([converts[i], temp_results[i]])
      for i in range(n_cnns) ]

    temp_results  = Concatenate()(temp_results) if not len(temp_results) == 1 else temp_results[0]
    # temp_results = back_lookup([converts[2], temp_results[0]])
    return temp_results

def get_model(wordMatrix, model_param=None) :
    if model_param is None :
        model_param = {'layer_num':1}
    if not 'pooling_mode' in model_param :
        model_param['pooling_mode'] = 'max'#'max'

    paras = get_para()

    """   inputs   """
    qu_ids    = Input(shape=(paras['qu_l'],), dtype='int32', name='qu')
    pre_ids   = Input(shape=(paras['pre_l'],), dtype='int32', name='pre')
    input_layers = [qu_ids, pre_ids]
    # (?, 50) (?, 20)

    """   embedding   """
    input_dim, output_dim = np.shape(wordMatrix)
    embedding = Embedding(input_dim, output_dim, weights=[wordMatrix], name='embedding')

    qu_words = embedding(qu_ids)
    pre_words = embedding(pre_ids)
    # (?, 50, 300) (?, 20, 300)

    """   index_inputs   """
    qu_convert1 = Input(shape=(paras['qu_sum1'],paras['qu_l']), dtype='float32', name='qu_convert1')
    qu_convert2 = Input(shape=(paras['qu_sum2'],paras['qu_l']), dtype='float32', name='qu_convert2')
    qu_convert3 = Input(shape=(paras['qu_sum3'],paras['qu_l']), dtype='float32', name='qu_convert3')
    pre_convert1 = Input(shape=(paras['pre_sum1'],paras['pre_l']), dtype='float32', name='pre_convert1')
    pre_convert2 = Input(shape=(paras['pre_sum2'],paras['pre_l']), dtype='float32', name='pre_convert2')
    pre_convert3 = Input(shape=(paras['pre_sum3'],paras['pre_l']), dtype='float32', name='pre_convert3')
    input_layers += [qu_convert1, qu_convert2, qu_convert3, pre_convert1, pre_convert2, pre_convert3] # 这里写成float只是算梯度什么的比较方便~

    qu_uig_group  = get_ig_inputs(1, paras['qu_i'][0], 'qu')
    qu_big_group  = get_ig_inputs(2, paras['qu_i'][1], 'qu')
    qu_tig_group  = get_ig_inputs(3, paras['qu_i'][2], 'qu')
    pre_uig_group = get_ig_inputs(1, paras['pre_i'][0], 'pre')
    pre_big_group = get_ig_inputs(2, paras['pre_i'][1], 'pre')
    pre_tig_group = get_ig_inputs(3, paras['pre_i'][2], 'pre')
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
    pre_results = one_layer(pre_igs, pre_words, cnn1, paras['pre_i'], converts=[pre_convert1, pre_convert2, pre_convert3], 
        pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses)
    # (?, 80, 1024)
    # (?, 160, 1024)
    
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
        pre_results_2 = one_layer(pre_igs, pre_results, cnn2, paras['pre_i'], converts=[pre_convert1, pre_convert2, pre_convert3], 
            active=None, pooling_mode = model_param['pooling_mode'], attentive_dense=attentive_denses_t)
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
    """单元测试"""

    # sys.path.append('..')
    # from basic import *
    # w2v = W2V('../../data/vectorsw300l20.all')
    # wordMatrix = w2v.getMatrix()
    # # (155841, 300)

    wordMatrix = np.random.random([155841, 300])

    model = get_model(wordMatrix)

    model.summary()





