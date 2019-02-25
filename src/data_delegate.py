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
from multiprocessing import Pool



class CData() :
    def __init__(self, data, default_size=8000) :
        if type(data) is str :
            data = json.load(codecs.open(data, 'r', 'utf-8'))#[:365]
        self.data = data # 原始读入数据

        self.last_deal_base =None # 上次做base的function
        self.dealed_all_base=None # 做过base之后的data
        self.dealed_all = None    # 处理后的data
        self.shuffle_seed = 42    # 随机数种子

        self.default_size = default_size # 默认的输出size

    @staticmethod
    def analisys_report(preprocess) :
        infor = preprocess.report()
        analisys = lambda x: [np.percentile(x,t) for t in range(0, 101, 10)] 
        analisys_large = lambda x: [np.percentile(x,t) for t in range(80, 101, 2)] 
        analisys_small = lambda x: [np.percentile(x,t) for t in range(0, 21, 2)] 
        keys = sorted(infor.keys())
        for k in keys :
            print(k, len(infor[k]))
            # print(analisys(infor[k]))
            print(analisys_large(infor[k]))
            # print(analisys_small(infor[k]))
            print('\n')
        
    def get_data(self) :
        return self.data

    def random_seed(self, seed=None) :
        if seed is None :
            self.shuffle_seed += 1
            random.seed(self.shuffle_seed)
        else :
            random.seed(seed)

    def deal_base(self, function=None, force=False, verbose=True) :
        if function is None :
            function = lambda x: x

        if any([force, self.dealed_all_base is None, not function is self.last_deal_base]) :
            self.last_deal_base = function

            to_deal = self.data 
            if verbose :
                to_deal = tqdm(to_deal)

            self.dealed_all_base = list(map(function, to_deal))

    def get_data_generator(self, group_padding_function, batch_size, deal_base_func=None, deal_function=None, size=None, 
            regenerate=True, shuffle_seed=None, shuffle=True, verbose=True) :
        """
        @size : None, 使用default的大小； False, 不考虑大小，直接返回全部数据~
        @deal_function : 数据处理的函数，
            输出格式：
            {'input':{dict field:np.array}, 'output':{dict field:np.array}}
            或:
            [list of {'input':{dict field:np.array}, 'output':{dict field:np.array}}]
            为从每组数据中得到的训练数据、测试数据~（从一个数据中可能得到多个pair之类的~）
        @deal_base_func : 数据基础处理函数，一般建议为引入外部资源（如词向量表）并全局保持不变~
        @group_padding_function : 对处理好的矩阵做后处理padding的函数~ 为None就不做~
            输入一个list of dict, 输出是 dict of matrix, 这里建议放一些比较慢的操作~
        @regenerate : 如果该对象以前被预处理过，是否重新生成~但不重新做deal_base_func
            建议为True，考虑到生成数据中其实也可能有一些采样相关的过程，而且一般并不慢~慢的东西建议放到deal_base_func中
        @shuffle : 在size不为false时，强烈建议其为true
                   在size为false时，为了保序，建议其为false
        @verbose : 是否显示处理数据的进度条~

        """
        if size is None :
            size = self.default_size

        """   数据处理   """
        if any([self.dealed_all is None, regenerate is True]) :
            # deal_base
            self.deal_base(function=deal_base_func, force=False, verbose=verbose)

            # deal function
            to_deal = self.dealed_all_base
            if verbose :
                to_deal = tqdm(to_deal)

            if deal_function is None :
                deal_function = lambda x: x

            self.dealed_all = list(map(deal_function, to_deal)) 
        data_all = [t for t in self.dealed_all] # deep copy

        """   shuffle 处理，在 size 不为false时必须进行~  """
        if all([not size is False, shuffle is False]) :
            print('warnning!!! size is not full, but shuffle is False!!')
        if all([size is False, shuffle is True]) :
            print('warnning!!! size is full, but shuffle is True!!')

        if shuffle :
            self.random_seed(shuffle_seed)
            random.shuffle(data_all)


        """  数量裁剪  """
        if not size is False :
            data_all = data_all[:size]
        data_to_return = [t for t in data_all]
        data_size = len(data_to_return)

        """
        list 的打破
        如果deal_function返回的是一组训练数据而不是一个的话，在这里进行恢复~
        """
        if type(data_to_return[0]) is list :
            data_to_return = [t for tt in data_to_return for t in tt]

            # # 如果打破list的话，重新做一次shuffle？
            # if shuffle :
            #     self.random_seed(shuffle_seed)
            #     random.shuffle(data_to_return)

        self.it_size = len(list(range(0, len(data_to_return), batch_size)))

        """   generator   """
        def generator(data_to_return, batch_size) :
            for i in range(0, len(data_to_return), batch_size) :
                data_selected = data_to_return[i:i+batch_size]
                inputs, outputs   = group_padding_function(data_selected)
                yield (inputs, outputs)

        return generator(data_to_return, batch_size)

        


if __name__ == '__main__':
    pass
    """单元测试"""

    from preprocess import CPreprocess
    preprocess = CPreprocess(w2v_path = '../data_vectors/vectorsw300l20.all')
    # data_test  = CData('../data/test_re_9413_segword.json')
    # data_train = CData('../data/train_re_14262_segword.json')
    # data_test  = CData('../data/test_re_9413_segmws.json')
    # data_train = CData('../data/train_re_14262_segmws.json')

    # data_test  = CData('../data/test_dbqa_5997_segword.json')
    # data_train = CData('../data/train_dbqa_8768_segword.json')
    # data_test  = CData('../data/test_dbqa_5997_segmws.json')
    # data_train = CData('../data/train_dbqa_8768_segmws.json')

    # train_generator = data_train.get_data_generator(
    #     preprocess.padding_nlpcckbqa_word, 64, 
    #     deal_base_func=preprocess.deal_nlpcckbqa_word_base,
    #     deal_function=preprocess.deal_nlpcckbqa_train,)

    # test_generator  = data_test.get_data_generator(
    #     preprocess.padding_nlpcckbqa_word, 64, 
    #     deal_base_func=preprocess.deal_nlpcckbqa_word_base,
    #     deal_function=preprocess.deal_nlpcckbqa_test, 
    #     size=False, shuffle=False)

    # train_generator = data_train.get_data_generator(
    #     preprocess.padding_nlpcckbqa_mws, 64, 
    #     deal_base_func=preprocess.deal_nlpcckbqa_mws_base,
    #     deal_function=preprocess.deal_nlpcckbqa_train,)

    # test_generator  = data_test.get_data_generator(
    #     preprocess.padding_nlpcckbqa_mws, 64, 
    #     deal_base_func=preprocess.deal_nlpcckbqa_mws_base,
    #     deal_function=preprocess.deal_nlpcckbqa_test, 
    #     size=False, shuffle=False)

    train_generator = data_train.get_data_generator(
        preprocess.padding_nlpccirqa_mws, 4, 
        deal_base_func=preprocess.deal_nlpccirqa_mws_base,
        deal_function=preprocess.deal_nlpccirqa_train,)
    test_generator  = data_test.get_data_generator(
        preprocess.padding_nlpccirqa_mws, 4, 
        deal_base_func=preprocess.deal_nlpccirqa_mws_base,
        deal_function=preprocess.deal_nlpccirqa_train, 
        size=False, shuffle=False)



    def check_generator(generator, infor) :
        print(infor)

        # for x, y in tqdm(generator) :
        #     pass
        # return None
        for x, y in tqdm(generator) :
            print(x)
            print(y)
            return None

        # t = [(x, y) for x, y in tqdm(generator)]
        t = [next(generator)]
        print('n_batch: %d'%len(t))
        # return None

        x0, y0 = t[0]
        xk = list(x0.keys())
        yk = list(y0.keys())

        print(y0)

        lens_all = [(len(xt[xk[0]]), len(yt[yk[0]])) for xt, yt in t]
        print('lens_all: ', lens_all)
        
        x0, y0 = t[0]

        print('x0:')
        for k in x0 :
            print(k, np.shape(x0[k]))
        print('y0:')
        for k in y0 :
            print(k, np.shape(y0[k]))

        print('\n')


    check_generator(train_generator, 'train_generator')
    check_generator(test_generator, 'test_generator')
    
    # CData.analisys_report(preprocess)


    # word_matrix = preprocess.get_w2v_matrix()
    # print('word_matrix', np.shape(word_matrix))





