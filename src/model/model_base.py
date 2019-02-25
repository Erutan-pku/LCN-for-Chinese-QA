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
from keras.layers import Input, Embedding, Activation, Dense, Flatten, Dropout, Lambda, Reshape, Add
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Concatenate, Multiply
import keras.backend as K

"""   loss functions   """
def my_loss_llh(y_true, y_pred) :
    #  y_true.shape : (?, ?) 
    #  y_pred.shape : (?, 1)
    """
    cross_entropy : 
    也许也是对数似然？
    - sum_i ( p_i * log(q_i) ) 其中，pi为真实i类别概率，qi为预测i类别概率~
    """

    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    loss = - ( y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred) )
    loss = tf.reduce_mean(loss)

    return loss

def count_similarity(qu_seq_1, pre_seq_1, pooling_mode='max') :
    myMaxPooling1D = Lambda(lambda x:tf.reduce_max(x, 1))
    assert pooling_mode in ['max', 'none']
    if pooling_mode == 'max' :
        qu_vec = myMaxPooling1D(qu_seq_1)
        pre_vec = myMaxPooling1D(pre_seq_1)
    elif pooling_mode == 'none' :
        qu_vec = qu_seq_1
        pre_vec = pre_seq_1
    # (?, 1024) (?, 1024)
    merge =  Multiply()([qu_vec, pre_vec])
    # (?, 1024)

    dense_merge = Dense(1024, activation='relu', name='dense_merge_1')
    dense_final = Dense(1, activation='sigmoid', name='label')
    merge = Dropout(0.5)(merge)
    merge = dense_merge(merge)
    merge = Dropout(0.5)(merge)
    # (?, 1024)

    label = dense_final(merge)
    # (?, 1)
    return label

def get_classify(test_seq, n_class) :
    assert type(n_class) is int

    dense_hidden = Dense(1024, activation='relu', name='dense_hidden_1')
    dense_final  = Dense(n_class, activation='softmax', name='label')

    myMaxPooling1D = Lambda(lambda x:tf.reduce_max(x, 1))

    hidden_layer = myMaxPooling1D(test_seq)
    hidden_layer = Dropout(0.5)(hidden_layer)
    hidden_layer = dense_hidden(hidden_layer)
    hidden_layer = Dropout(0.5)(hidden_layer)

    label = dense_final(hidden_layer)

    return label



