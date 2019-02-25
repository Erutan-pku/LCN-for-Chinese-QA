#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import re
import os
import json
import time
import codecs
import random
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append('..')
from basic import *

codecs_out = lambda x : codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)

char_num=set('abcdefghjiklmnopqrstuvwxyz1234567890.')

""" ===== ===== ===== ===== ===== ===== """


def get_nlpccdbqa_word(cws, input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    for dt in data_all :
        qu_words = cws.jieba_cws2(dt['qu'])
        sent_words = [cws.jieba_cws2(t) for t in dt['sents']]
        
        dt['sents'] = sent_words
        dt['qu'] = qu_words
        
    json_dump(data, output_path)

def analisys_words() :
    # analisys = lambda x: [np.percentile(x,t) for t in range(0, 101, 2)] # + [min(x), max(x)]

    data = json_load('../../data/test_dbqa_5997_segword.json')+json_load('../../data/train_dbqa_8768_segword.json')
    all_keys = ['len_sent', 'len_qu']
    counts={k:[] for k in all_keys}

    for dt in tqdm(data) :
        counts['len_qu'].append(len(dt['qu']))
        counts['len_sent'] += list(map(len, dt['sents']))

    for k in all_keys :
        print(k, len(counts[k]), analisys(counts[k]))

""" ===== ===== ===== ===== ===== ===== """

from get_nlpcckbqa import post_deal

def get_nlpccdbqa_wordBank(cws, input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    for idt, dt in enumerate(data_all) :
        """   qu_cut   """
        dt['qu']=dt['qu'].strip()
        dt['sents']=[t.strip() for t in dt['sents']]
        all_qu_words = cws.MWS(dt['qu'], index=True)
 
        """   pre_cut   """
        all_sents_words = [cws.MWS(t, index=True) for t in dt['sents']]

        """   post_deal   """
        all_qu_words = post_deal(all_qu_words, dt['qu'])
        all_sents_words = [post_deal(t, dt['sents'][i]) for i, t in enumerate(all_sents_words)]

        """   update   """
        dt['sents'] = all_sents_words
        dt['qu'] = all_qu_words
        
    json_dump(data, output_path)

from get_nlpcckbqa import ana_word_seq

def analisys_mws() :
    # analisys = lambda x: [np.percentile(x,t) for t in range(0, 101, 10)] 
    analisys = lambda x: [np.percentile(x,t) for t in range(80, 101, 2)] 

    data = json_load('../../data/test_dbqa_5997_segmws.json')+json_load('../../data/train_dbqa_8768_segmws.json')
    all_keys = ['len_sent', 'len_qu']
    for i in range(4) :
        for t in ['qu', 'sent'] :
            all_keys.append('n%d_%s'%(i+1, t))
    # 'n1_sent' ~ 'n4_sent', 'n3_qu'...
    counts={k:[] for k in all_keys}

    for dt in tqdm(data) :
        ana_ret = ana_word_seq(dt['qu'])
        kt = 'qu'
        counts['len_%s'%(kt)].append(ana_ret['len'])
        for i in range(4) :
            counts['n%d_%s'%(i+1, kt)].append(ana_ret['len_max%d'%(i+1)])

        kt = 'sent'
        for ana_ret in list(map(ana_word_seq, dt['sents'])) :
            counts['len_%s'%(kt)].append(ana_ret['len'])
            for i in range(4) :
                counts['n%d_%s'%(i+1, kt)].append(ana_ret['len_max%d'%(i+1)])

    for k in all_keys :
        print(k, len(counts[k]), analisys(counts[k]))

    """
    这一组是80~100分位点之间，每 2% 打点的结果~ 
    len_sent 304293 [83.0, 87.0, 92.0, 98.0, 105.0, 114.0, 124.0, 138.0, 160.0, 200.0, 8630.0]
    len_qu 14765 [29.0, 29.0, 31.0, 32.0, 33.0, 34.0, 36.0, 38.0, 42.0, 48.0, 106.0]
    n1_qu 14765 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    n1_sent 304293 [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    n2_qu 14765 [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0]
    n2_sent 304293 [3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 9.0]
    n3_qu 14765 [4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 9.0, 17.0]
    n3_sent 304293 [6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0, 11.0, 81.0]
    n4_qu 14765 [7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 9.0, 10.0, 11.0, 14.0, 43.0]
    n4_sent 304293 [10.0, 11.0, 12.0, 12.0, 13.0, 15.0, 16.0, 18.0, 20.0, 25.0, 729.0]
    """

""" ===== ===== ===== ===== ===== ===== """

def convert_word_mws(input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    deal_seq = lambda x :[[t, i, i+1] for i, t in enumerate(x)] if not len(x) == 0 else [["<NW_oi>", 0, 4]]

    data_used = []
    for dt in data_all :
        dt['qu'] = deal_seq(dt['qu'])
        dt['sents'] = list(map(deal_seq, dt['sents']))
        data_used.append(dt)

    json_dump(data_used, output_path)

def convert_character_real(input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    deal_seq = lambda x :[[t, i, i+1] for i, t in enumerate(x)] if not len(x) == 0 else [["<NW_oi>", 0, 4]]
    deal_char = lambda x : deal_seq(''.join(x))

    data_used = []
    for dt in data_all :
        dt['qu'] = deal_char(dt['qu'])
        dt['sents'] = list(map(deal_char, dt['sents']))
        data_used.append(dt)

    json_dump(data_used, output_path)   




""" ===== ===== ===== ===== ===== ===== """

from get_nlpcckbqa import get_chatacter, get_longest_chain, get_shortest_chain

def convert_character_mws(input_path, output_path, verbose=True, func=get_chatacter) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    data_used = []
    for dt in data_all :
        dt['qu'] = func(dt['qu'])
        dt['sents'] = list(map(func, dt['sents']))
        data_used.append(dt)

    json_dump(data_used, output_path)   


""" ===== ===== ===== ===== ===== ===== """

if __name__ == '__main__':
    cws = CWS('../../data_vectors/WordBank_v')

    get_nlpccdbqa_word(cws, '../../data/test_dbqa_5997.json', '../../data/test_dbqa_5997_segword.json')
    get_nlpccdbqa_word(cws, '../../data/train_dbqa_8768.json', '../../data/train_dbqa_8768_segword.json')
    
    convert_character_real('../../data/test_dbqa_5997_segword.json', '../../data/test_dbqa_5997_segchar_real.json')
    convert_character_real('../../data/train_dbqa_8768_segword.json', '../../data/train_dbqa_8768_segchar_real.json')

    get_nlpccdbqa_wordBank(cws, '../../data/test_dbqa_5997.json', '../../data/test_dbqa_5997_segmws.json')
    get_nlpccdbqa_wordBank(cws, '../../data/train_dbqa_8768.json', '../../data/train_dbqa_8768_segmws.json')

    """
    对nlpcc dbqa 2016 数据做分词
    分词方式为：
        问题：jieba 之后 MM 细粒度处理~
        句子：jieba 之后 MM 细粒度处理~
    输出文件为：
        ../../data/test_dbqa_5997_segword.json
        ../../data/train_dbqa_8768_segword.json
    """

    