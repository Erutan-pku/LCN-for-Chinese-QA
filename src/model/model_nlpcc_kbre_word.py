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
from keras.layers import Input, Embedding, Activation, Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Add
from keras.layers import Concatenate, Multiply
import keras.backend as K

from model_base import *

def get_model(wordMatrix, model_param=None) :
    if model_param is None :
        model_param = {'layer_num':1}
    len_qu  = 50
    len_pre = 20

    """   inputs   """
    qu_ids    = Input(shape=(len_qu,), dtype='int32', name='qu')
    pre_ids   = Input(shape=(len_pre,), dtype='int32', name='pre')
    # (?, 50) (?, 20)

    """   embedding   """
    input_dim, output_dim = np.shape(wordMatrix)
    embedding = Embedding(input_dim, output_dim, weights=[wordMatrix], name='embedding')

    qu_seq = embedding(qu_ids)
    pre_seq = embedding(pre_ids)
    # (?, 50, 300) (?, 20, 300)

    """   convolution   """
    cnn1_0 = Conv1D(256, 1, padding='same', name='cnn1_0')
    cnn1_1 = Conv1D(512, 2, padding='same', name='cnn1_1')
    cnn1_2 = Conv1D(256, 3, padding='same', name='cnn1_2')
    cnn1 = lambda x : Concatenate()([cnn_t(x) for cnn_t in [cnn1_0, cnn1_1, cnn1_2]])

    qu_seq_1   = cnn1(qu_seq)
    pre_seq_1  = cnn1(pre_seq)
    qu_seq_1   = Activation('relu')(qu_seq_1)
    pre_seq_1  = Activation('relu')(pre_seq_1)
    # (?, 50, 1024) (?, 20, 1024)

    for i in range(model_param['layer_num']-1) :
        cnn2_0 = Conv1D(256, 1, padding='same', name='cnn%d_0'%(i+2))
        cnn2_1 = Conv1D(512, 2, padding='same', name='cnn%d_1'%(i+2))
        cnn2_2 = Conv1D(256, 3, padding='same', name='cnn%d_2'%(i+2))
        cnn2 = lambda x : Concatenate()([cnn_t(x) for cnn_t in [cnn2_0, cnn2_1, cnn2_2]])

        qu_seq_2 = cnn2(qu_seq_1)
        pre_seq_2 = cnn2(pre_seq_1)
        # (?, 50, 1024)
        # (?, 20, 1024)

        qu_seq_2 = Activation('relu')(Add()([qu_seq_1, qu_seq_2]))
        pre_seq_2 = Activation('relu')(Add()([pre_seq_1, pre_seq_2]))
        qu_seq_1 = qu_seq_2
        pre_seq_1 = pre_seq_2

    """   merge   """
    label = count_similarity(qu_seq_1, pre_seq_1)
    # (?, 1)

    model = Model(inputs=[qu_ids, pre_ids], outputs=[label])
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




