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


def get_nlpcckbqa_word(cws, input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    for dt in data_all :
        qu_preEnt = dt['qu'][:dt['men_start']]
        qu_posEnt = dt['qu'][dt['men_start']+len(dt['sub_men']):]
        qu_words = cws.RMM(qu_preEnt) + ['<Entity>'] + cws.MM(qu_posEnt)

        pre_words = [cws.jieba_cws2(t) for t in dt['pres']]
        
        dt['labels'] = [1 if t == dt['gold_triple'][1] else 0 for t in dt['pres']]
        dt['pres'] = pre_words
        dt['qu'] = qu_words
        
    json_dump(data, output_path)

""" ===== ===== ===== ===== ===== ===== """

def post_deal(word_seq, str_seq) :
    # print(word_seq, str_seq)
    t1 =len(word_seq)#print(word_seq)
    if len(word_seq) == 0 :
        # 这都是纯外文词汇的成分，当成个未登录词吧~
        return [['<NW_oi>', 0, 4]]

    start_ls = set([t[1] for t in word_seq])
    ent_ls   = set([t[2] for t in word_seq])
    start_l  = min(start_ls)
    end_l    = max(ent_ls)

    if not all([start_l==0, len(str_seq) == end_l]) :
        pass
        # 这部分的特例见附录，无视掉差的头尾，直接做吧~


    """   如果a和a+1都是数字或字母，中间就不应该分割~   不过如果词表中没词，那也必须要分~   """
    no_split_points = [i for i in range(len(str_seq)-1) if all([str_seq[i]in char_num, str_seq[i+1]in char_num])]

    word_seq_new = []
    for t in word_seq :
        if not any([t[1]-1 in no_split_points, t[2]-1 in no_split_points]) :
            word_seq_new.append(t)
            continue

        t_cover = [tt for tt in word_seq if all([tt[1]<=t[1], tt[2]>=t[2], tt[2]-tt[1]>t[2]-t[1]])]
        if len(t_cover) == 0 :
            # print (t_cover, t, str_seq)
            word_seq_new.append(t)

    """   处理中间流间断的情况，加入UNK   """
    unk_dict = {1:'<NW_uni>', 2:'<NW_bi>', 3:'<NW_tri>'}
    get_unk = lambda l: unk_dict[l] if l in unk_dict else '<NW_oi>'
    mark=False    
    while True :
        start_ls = set([t[1] for t in word_seq])
        end_ls   = set([t[2] for t in word_seq])
        # print(start_ls, start_l)
        # print(end_ls, end_l)

        no_end   = start_ls-end_ls-set([start_l])
        no_start = end_ls-start_ls-set([end_l])
        # print(no_end)
        # print(no_start)
        
        if len(no_end) == 0 and len(no_start) == 0 :
            break
        mark = True

        if len(no_end) > 0:
            t_start = no_end.pop()
            last_end = max([t[2] for t in word_seq if t[2]<t_start]+[start_l])
            temp_word = str_seq[last_end:t_start]
            if re.match(' +', temp_word) :
                lent = len(temp_word)
                str_seq = str_seq[:last_end]+str_seq[t_start:]
                for t in word_seq :
                    if t[1] > last_end :
                        t[1] -= lent
                    if t[2] > last_end :
                        t[2] -= lent
                end_l -= lent
            else :
                unk_word = get_unk(len(temp_word))
                word_seq.append([unk_word, last_end, t_start])
            continue

        if len(no_start) > 0:
            t_end = no_start.pop()
            next_start = min([t[1] for t in word_seq if t[1]>=t_end]+[end_l])
            temp_word = str_seq[t_end:next_start]
            if re.match(' +', temp_word) :
                lent = len(temp_word)
                str_seq = str_seq[:t_end]+str_seq[next_start:]
                for t in word_seq :
                    if t[1] > t_end :
                        t[1] -= lent
                    if t[2] > t_end :
                        t[2] -= lent
                end_l -= lent
            else :
                unk_word = get_unk(len(temp_word))
                word_seq.append([unk_word, t_end, next_start])
            continue
            
    # if mark :
    #     for t in str_seq :
    #         print(t)
    #     for t in word_seq :
    #         print(t[0], t[1], t[2])
    return word_seq

def get_nlpcckbqa_wordBank(cws, input_path, output_path, verbose=True) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    for idt, dt in enumerate(data_all) :
        # if not dt['id'] == 372 :
        #     continue
        """   qu_cut   """
        dt['pres'] = [t.strip() for t in dt['pres']]
        qu_preEnt = dt['qu'][:dt['men_start']].strip()
        qu_posEnt = dt['qu'][dt['men_start']+len(dt['sub_men']):].strip()
        qu_noEnt  = qu_preEnt+' '+qu_posEnt

        qu_preEnt_ws = cws.MWS(qu_preEnt, index=True)
        qu_posEnt_ws = cws.MWS(qu_posEnt, index=True)
        for t in qu_posEnt_ws :
            t[1] += dt['men_start'] + 1
            t[2] += dt['men_start'] + 1
        
        all_qu_words = [['<Entity>', dt['men_start'], dt['men_start']+1]]
        all_qu_words += qu_preEnt_ws+qu_posEnt_ws

        """   pre_cut   """
        all_pres_words = [cws.MWS(t.strip(), index=True) for t in dt['pres']]

        """   post_deal   """
        all_qu_words = post_deal(all_qu_words, qu_noEnt)
        all_pres_words = [post_deal(t, dt['pres'][i]) for i, t in enumerate(all_pres_words)]

        """   update   """
        dt['labels'] = [1 if t == dt['gold_triple'][1] else 0 for t in dt['pres']]
        dt['pres'] = all_pres_words
        dt['qu'] = all_qu_words
        


    json_dump(data, output_path)

def ana_word_seq(word_seq) :
    ret = {}
    ret['len'] = len(word_seq)
    
    p_head_dict = {}
    p_tail_dict = {}
    for t in word_seq :
        if not t[1] in p_head_dict :
            p_head_dict[t[1]] = []
        if not t[2] in p_tail_dict :
            p_tail_dict[t[2]] = []
        p_head_dict[t[1]].append(t)
        p_tail_dict[t[2]].append(t)
    
    all_node_keys = set([tt for t in word_seq for tt in [t[1], t[2]]])
    all_node_keys = sorted(list(all_node_keys))
    all_node = {idt:{'id':idt} for idt in all_node_keys}
    for k in all_node :
        t = all_node[k]
        t['start'] = p_head_dict.get(t['id'], [])
        t['end']   = p_tail_dict.get(t['id'], [])

    head_1_num = [[[t]] for t in word_seq]
    head_2_num = [[tt+[m] for tt in t for m in all_node[tt[-1][2]]['start']] 
        for t in head_1_num ]
    head_3_num = [[tt+[m] for tt in t for m in all_node[tt[-1][2]]['start']] 
        for t in head_2_num ]
    head_4_num = [[tt+[m] for tt in t for m in all_node[tt[-1][2]]['start']] 
        for t in head_3_num ]
    for i, t in enumerate([head_1_num, head_2_num, head_3_num, head_4_num]) :
        ret['len_max%d'%(i+1)] = max(list(map(len, t)))
    return ret

def analisys_mws() :
    analisys = lambda x: [np.percentile(x,t) for t in range(0, 101, 10)] # + [min(x), max(x)]

    data = json_load('../../data/test_re_9413_segmws.json')+json_load('../../data/train_re_14262_segmws.json')
    all_keys = ['len_pre', 'len_qu']
    for i in range(4) :
        for t in ['qu', 'pre'] :
            all_keys.append('n%d_%s'%(i+1, t))
    # 'n1_pre' ~ 'n4_pre', 'n3_qu'...
    counts={k:[] for k in all_keys}

    for dt in tqdm(data) :
        ana_ret = ana_word_seq(dt['qu'])
        kt = 'qu'
        counts['len_%s'%(kt)].append(ana_ret['len'])
        for i in range(4) :
            counts['n%d_%s'%(i+1, kt)].append(ana_ret['len_max%d'%(i+1)])

        kt = 'pre'
        for ana_ret in list(map(ana_word_seq, dt['pres'])) :
            counts['len_%s'%(kt)].append(ana_ret['len'])
            for i in range(4) :
                counts['n%d_%s'%(i+1, kt)].append(ana_ret['len_max%d'%(i+1)])

    for k in all_keys :
        print(k, len(counts[k]), analisys(counts[k]))

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
        dt['pres'] = list(map(deal_seq, dt['pres']))
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
        dt['qu'] = [t if not t == '<Entity>' else 'E' for t in dt['qu']]
        dt['qu'] = deal_seq(''.join(dt['qu']))
        for token in dt['qu'] :
            if token[0] == 'E' :
                token[0] = '<Entity>'
        dt['pres'] = list(map(deal_char, dt['pres']))
        data_used.append(dt)

    json_dump(data_used, output_path)   

""" ===== ===== ===== ===== ===== ===== """

def get_p_dict(word_seq) :
    """   建立基于位置的索引   """
    p_head_dict = {}
    p_tail_dict = {}
    for t in word_seq :
        if not t[1] in p_head_dict :
            p_head_dict[t[1]] = []
        if not t[2] in p_tail_dict :
            p_tail_dict[t[2]] = []
        p_head_dict[t[1]].append(t)
        p_tail_dict[t[2]].append(t)
    # 字典，储存了以 k（id） 为开头或结尾的token list
    # 预处理保证了，每个开头位置都有词结尾，每个结尾位置都有词开头~

    return p_head_dict, p_tail_dict

def get_chatacter(word_seq) :
    """贪心法，获取最长子链"""
    p_head_dict, p_tail_dict = get_p_dict(word_seq)
    
    min_head = min(p_head_dict.keys())
    max_tail = max(p_tail_dict.keys())
    temp_tail = min_head
    choose_t = []
    while not temp_tail == max_tail :
        word_group = p_head_dict[temp_tail]
        min_end = min([t[2] for t in word_group])
        next_word = [t for t in word_group if t[2]==min_end][0]
        
        choose_t.append(next_word)
        temp_tail = next_word[2]
    return choose_t

def get_longest_chain(word_seq) :
    """动态规划，获取最长子链"""
    p_head_dict, p_tail_dict = get_p_dict(word_seq)
    p_head_list = sorted(p_head_dict.keys())
    max_tail = max(p_tail_dict.keys())

    l_tail = {} # [length, [list of [token]]]
    for pt in p_head_list :
        if not pt in l_tail :
            prev_len, prev_list = 0, [[]]
        else :
            prev_len, prev_list = l_tail[pt]

        for token in p_head_dict[pt] :
            tail_pt = token[2]
            if not tail_pt in l_tail :
                l_tail[tail_pt] = [prev_len+1, [t+[token] for t in prev_list]]
            elif prev_len + 1 == l_tail[tail_pt][0] :
                l_tail[tail_pt][1] += [t+[token] for t in prev_list]
            elif prev_len + 1 > l_tail[tail_pt][0] :
                l_tail[tail_pt] = [prev_len+1, [t+[token] for t in prev_list]]

    return l_tail[max_tail][1][0]

def get_shortest_chain(word_seq) :
    """动态规划，获取最短子链"""
    p_head_dict, p_tail_dict = get_p_dict(word_seq)
    p_head_list = sorted(p_head_dict.keys())
    max_tail = max(p_tail_dict.keys())

    l_tail = {} # [length, [list of [token]]]
    for pt in p_head_list :
        if not pt in l_tail :
            prev_len, prev_list = 0, [[]]
        else :
            prev_len, prev_list = l_tail[pt]
        prev_list = prev_list[:1]

        for token in p_head_dict[pt] :
            tail_pt = token[2]
            if not tail_pt in l_tail :
                l_tail[tail_pt] = [prev_len+1, [t+[token] for t in prev_list]]
            elif prev_len + 1 == l_tail[tail_pt][0] :
                l_tail[tail_pt][1] += [t+[token] for t in prev_list]
            elif prev_len + 1 < l_tail[tail_pt][0] :
                l_tail[tail_pt] = [prev_len+1, [t+[token] for t in prev_list]]

    return l_tail[max_tail][1][0]

def convert_character_mws(input_path, output_path, verbose=True, func=get_chatacter) :
    data = json_load(input_path)

    data_all = data
    if verbose is True :
        data_all = tqdm(data_all)

    data_used = []
    for dt in data_all :
        dt['qu'] = func(dt['qu'])
        dt['pres'] = list(map(func, dt['pres']))
        data_used.append(dt)

    json_dump(data_used, output_path)        


def check_word_seq(seq1, seq2) :
    # # for t in seq1+seq2 :
    # for t in seq1:
    #     if len(t[0]) > 1 and not t[0][0]=='<':
    #         # print(seq1)
    #         print(t[0])

    str_1 = ' '.join([t[0] for t in seq1])
    str_2 = ' '.join([t[0] for t in seq2])
    if not str_1 == str_2 :
        print(str_1)
        print(str_2)
        return True
    return False

def analisys_chain(input_1, input_2, keys) :
    data1 = json_load(input_1)
    data2 = json_load(input_2)

    assert len(data1) == len(data2)
    lent = len(data1)
    for i in tqdm(list(range(lent))) :
        dt1 = data1[i]
        dt2 = data2[i]
        id_t = dt1['id']

        mark = False
        for key_type, key in keys :
            if key_type == 'str' :
                mark|=check_word_seq(dt1[key], dt2[key])
            elif key_type == 'list' :
                for j in range(len(dt1[key])) :
                    mark|=check_word_seq(dt1[key][j], dt2[key][j])

        if mark :
            print(id_t+'\n')

"""
160 [[142, 0, 1], [11503, 1, 2], [5937, 2, 3], [11503, 3, 4], [5937, 4, 5], [6010, 5, 6], 
[16, 6, 7], [108, 7, 8], [2, 8, 9], [4873, 9, 10], [29662, 9, 11], [155838, 10, 12], [155840, 12, 12]]
"""


""" ===== ===== ===== ===== ===== ===== """

if __name__ == '__main__':
    cws = CWS('../../data_vectors/WordBank_v')

    get_nlpcckbqa_word(cws, '../../data/test_re_9413.json', '../../data/test_re_9413_segword.json')
    get_nlpcckbqa_word(cws, '../../data/train_re_14262.json', '../../data/train_re_14262_segword.json')
    
    convert_character_real('../../data/test_re_9413_segword.json', '../../data/test_re_9413_segchar_real.json')
    convert_character_real('../../data/train_re_14262_segword.json', '../../data/train_re_14262_segchar_real.json')

    get_nlpcckbqa_wordBank(cws, '../../data/test_re_9413.json', '../../data/test_re_9413_segmws.json')
    get_nlpcckbqa_wordBank(cws, '../../data/train_re_14262.json', '../../data/train_re_14262_segmws.json')

    """
    对nlpcc kbqa 2016 数据做分词
    分词方式为：
        问题：实体 -> <Entity> ，之前RMM，之后MM
        谓词：jieba 之后 MM 细粒度处理~
    输出文件为：
        ../../data/test_re_9413_segword.json
        ../../data/train_re_14262_segword.json
    """

