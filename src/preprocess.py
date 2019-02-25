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

from basic import *

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class CPreprocess() :
    def __init__(self, w2v_path) :
        self.w2v = W2V(w2v_path)
        self.random_seed = 42

        self.__get_seed(np.random.seed)
        vt = np.random.random(self.w2v.getVecLen()) * 2. - 1.
        self.w2v.addWord('<Entity>', vt)

        # 可以用来存一些日志类信息
        self.log_info = {}

        # 可以用来存一些信息，不过调用时有过时的危险~
        self.pad_info = None

    def __get_seed(self, func=None) :
        t = self.random_seed
        self.random_seed += 1
        if not func is None :
            func(t)
        return t

    def get_w2v_matrix(self) :
        return self.w2v.getMatrix()
    
    def report(self, verbose=False) :
        if verbose :
            print(json.dumps(self.log_info, indent=2))
        return self.log_info

    """对nlpcckbqa word 的预处理"""
    def deal_nlpcckbqa_word_base(self, x) :
        output = {}
        get_ids = lambda x : [self.w2v.getWordID(t) for t in x]

        output['qu_ids'] = get_ids(x['qu'])
        output['pre_ids'] = [get_ids(t) for t in x['pres']]
        output['pre_labels'] = x['labels']

        """
        print (output['qu_ids'])
        print (output['pre_ids'])
        print (output['pre_labels'])
        [155840, 198, 690, 1, 31, 3, 1457]
        [[3879], [40353], [29904], [70, 36], [177], [508, 54], [299, 175], [1782], [8], [31], [388, 1555], [51, 54], [4444], [123], [245, 101], [205, 54], [1282], [4119, 225]]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        """
    
        return output
    def deal_nlpcckbqa_train(self, x_base, num=5) :
        pre_len = len(x_base['pre_ids'])

        self.__get_seed(random.seed)
        index = list(range(pre_len))
        random.shuffle(index)

        true_id = [i for i in range(pre_len) if x_base['pre_labels'][i] == 1][0]
        if not true_id in index[:num] :
            index = [true_id]+index
        index = index[:num]
        output = [{'input':{'qu':x_base['qu_ids'], 
                   'pre':x_base['pre_ids'][idt]}, 
                   'output':{'label':x_base['pre_labels'][idt]}}
            for idt in index]

        return output
    def deal_nlpcckbqa_test(self, x_base) :
        output = [{'input':{'qu':x_base['qu_ids'], 
                   'pre':x_base['pre_ids'][idt]}, 
                   'output':{'label':x_base['pre_labels'][idt]}}
            for idt in range(len(x_base['pre_ids']))]

        return output
    def padding_nlpcckbqa_word(self, xs, len_qu=50, len_pre=20) :
        """   返回字典构建   """
        outputs = [t['output'] for t in xs]
        inputs  = [t['input']  for t in xs]
        output_keys = outputs[0].keys()
        input_keys  = inputs[0].keys()
        outputs = {key:np.array([t[key] for t in outputs])
            for key in output_keys}
        inputs  = {key:np.array([t[key] for t in inputs])
            for key in input_keys}

        """   padding   """
        head_words = np.array([0, 0, 0, 0])
        for key in ['qu', 'pre'] :
            inputs[key] = list(inputs[key])
            matrix = inputs[key]
            for i, line in enumerate(matrix) :
                line = np.array(line)
                t= np.concatenate((head_words, line))
                matrix[i] = t 

        inputs['qu']  = pad_sequences(inputs['qu'], maxlen=len_qu, padding='post', truncating='post')
        inputs['pre'] = pad_sequences(inputs['pre'], maxlen=len_pre, padding='post', truncating='post')

        return inputs, outputs

    """对nlpcckbqa mws 的预处理"""
    def deal_lattice_base(self, word_seq, max_len) :
        # print(max_len, word_seq)
        """
        这里的base，只做最基础的处理
            (转id，统计每个节点的n-gram列表，生成n-gram索引)
        不做padding，不做分堆，不做重排列
        """

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

        """   idseq, 为原始序列   """
        # 最开始要补一个零，因为要保证 idseq[0] 是pad，这个特性后面会利用~
        idseq = ([0]+[t[0] for t in word_seq])[:max_len]
        for i, t in enumerate(word_seq) :
            pt = i+1
            t.append(pt)
            if pt < max_len :
                assert idseq[pt] == t[0]
            else :
                # 如果碰巧超长了，这个位置代表的词一律视为pad，这个词在原序列中的所在位置也一律视为0
                # 不过这个主要是为 p_head_dict & p_tail_dict 中的token服务的吧~逼近word_seq是会被剪裁的~

                t[0] = 0 # 这个是这个词的id
                t[-1] = 0 # 这个是这个词对应在idseq中的位置~
        # 剪裁 word_seq，考虑到我们能够处理的token只有在idseq中有表示，且中心词不是pad的位置~
        word_seq = [t for t in word_seq if not t[-1] == 0]
        """
        事实上只能处理max_len-1个token~
        idseq:      [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49]
        word_seq:   [[155841, 0, 1, 1], [199, 1, 2, 2], [102, 2, 3, 3], [691, 2, 4, 4], 
            [204, 3, 4, 5], [2, 4, 5, 6], [213, 5, 6, 7], [71, 5, 7, 8], [9, 5, 8, 9], 
            [158, 6, 7, 10], [1170, 7, 8, 11], [4, 8, 9, 12], [859, 9, 10, 13], 
            [5506, 9, 11, 14], [49, 10, 11, 15]]
        idseq 为词ID序列，长度不超过max_len~过长则截断~
        word_seq 为lattice序列，内容为  (词id，首位置，尾位置，在idseq中的位置)
            如果词处于被截断的位置，则改词的词id为<pad>，对应在idseq的位置为0.
                （这主要只对p_head_dict & p_tail_dict两个字典生效，因为引用之类的）
            而word_seq只会保留中心词在idseq当中的内容，为后续真实的处理中心~
        后续会对每个word_seq的词计算周边的n-gram信息~
        """


        """
        在 head & tail dict 中手工做padding~
        在整个lattice的开头和结尾，补上 pad_num 个padding，token id=0, 对应idseq中的位置也是0
        """
        min_start = min([t[1] for t in word_seq])
        max_end   = max([t[2] for t in word_seq])
        pad_num = 2
        for i in range(pad_num) :
            p_head_dict[max_end+i] = [[0, max_end+i, max_end+i+1, 0]]
            p_tail_dict[min_start-i] = [[0, min_start-i-1, min_start-i, 0]]

        # word_seq, p_head_dict, p_tail_dict
        # [[155841, 0, 1, 1], [199, 1, 2, 2], [102, 2, 3, 3], [691, 2, 4, 4], [204, 3, 4, 5], [2, 4, 5, 6], [213, 5, 6, 7], [71, 5, 7, 8], [9, 5, 8, 9], [158, 6, 7, 10], [1170, 7, 8, 11], [4, 8, 9, 12], [859, 9, 10, 13], [5506, 9, 11, 14], [49, 10, 11, 15]]
        # {0: [[155841, 0, 1, 1]], 1: [[199, 1, 2, 2]], 2: [[102, 2, 3, 3], [691, 2, 4, 4]], 3: [[204, 3, 4, 5]], 4: [[2, 4, 5, 6]], 5: [[213, 5, 6, 7], [71, 5, 7, 8], [9, 5, 8, 9]], 6: [[158, 6, 7, 10]], 7: [[1170, 7, 8, 11]], 8: [[4, 8, 9, 12]], 9: [[859, 9, 10, 13], [5506, 9, 11, 14]], 10: [[49, 10, 11, 15]], 11: [[0, 11, 12, 0]], 12: [[0, 12, 13, 0]]}
        # {0: [[0, -1, 0, 0]], 1: [[155841, 0, 1, 1]], 2: [[199, 1, 2, 2]], 3: [[102, 2, 3, 3]], 4: [[691, 2, 4, 4], [204, 3, 4, 5]], 5: [[2, 4, 5, 6]], 6: [[213, 5, 6, 7]], 7: [[71, 5, 7, 8], [158, 6, 7, 10]], 8: [[9, 5, 8, 9], [1170, 7, 8, 11]], 9: [[4, 8, 9, 12]], 10: [[859, 9, 10, 13]], 11: [[5506, 9, 11, 14], [49, 10, 11, 15]], -1: [[0, -2, -1, 0]]}

        """   生成n-gram 索引表   """
        unigrams  = [[[t[-1]]] for t in word_seq]
        bigrams   = [[[t[-1], t1[-1]] for t1 in p_head_dict[t[2]]] for t in word_seq]
        trigrams  = [[[t0[-1], t[-1], t1[-1]] for t1 in p_head_dict[t[2]] for t0 in p_tail_dict[t[1]]] for t in word_seq]

        # print map(len, [unigrams, bigrams, trigrams])
        # print map(np.shape, unigrams)
        # print map(np.shape, bigrams)
        # print map(np.shape, trigrams)
        # (15, 1, 1)
        # [(1, 2), (2, 2), (1, 2), (1, 2), (1, 2), (3, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        # [(1, 3), (2, 3), (1, 3), (1, 3), (1, 3), (6, 3), (1, 3), (1, 3), (1, 3), (1, 3), (2, 3), (4, 3), (1, 3), (1, 3), (1, 3)]

        """   返回值   """
        to_ret = {
            'seq' : idseq, 
            'uig' : unigrams, 
            'big' : bigrams, 
            'tig' : trigrams, 
        }

        return to_ret
    def deal_nlpcckbqa_mws_base(self, x, len_qu=50, len_pre=20) :
        output = {}
        """   转为ID   """
        get_ids = lambda x : [[self.w2v.getWordID(t[0]), t[1], t[2]] for t in x]

        output['qu_ids'] = get_ids(x['qu'])
        output['pre_ids'] = [get_ids(t) for t in x['pres']]

        qu_deals   = self.deal_lattice_base(output['qu_ids'], max_len=len_qu)
        pres_deals = [self.deal_lattice_base(t, max_len=len_pre) for t in output['pre_ids']]
        output['qu_ids'] = qu_deals
        output['pre_ids'] = pres_deals

        output['pre_labels'] = x['labels']

        """
        print (output['qu_ids'])
        print (output['pre_ids'])
        print (output['pre_labels'])
        [[155841, 0, 1], [199, 1, 2], [102, 2, 3], [691, 2, 4], [204, 3, 4], [2, 4, 5], [213, 5, 6], [71, 5, 7], [9, 5, 8], [158, 6, 7], [1170, 7, 8], [4, 8, 9], [859, 9, 10], [5506, 9, 11], [49, 10, 11]]
        [[[1226, 0, 1], [3880, 0, 2], [4223, 1, 2]], [[4885, 0, 1], [21775, 0, 2], [40354, 0, 4], [4276, 1, 2], [62040, 1, 3], [3856, 2, 3], [45976, 2, 4], [3511, 3, 4]], [[60, 0, 1], [29905, 0, 3], [2053, 1, 2], [958, 1, 3], [1050, 2, 3]], [[213, 0, 1], [71, 0, 2], [158, 1, 2], [183, 2, 3], [37, 2, 4], [486, 3, 4]], [[1298, 0, 1], [178, 0, 2], [18368, 1, 2]], [[4962, 0, 1], [509, 0, 2], [204, 1, 2], [1283, 1, 3], [55, 2, 3]], [[300, 0, 1], [176, 1, 2]], [[1225, 0, 1], [1783, 0, 2], [55, 1, 2]], [[213, 0, 1], [71, 0, 2], [9, 0, 3], [158, 1, 2], [1170, 2, 3]], [[395, 0, 1], [32, 0, 2], [226, 1, 2]], [[12543, 0, 1], [389, 0, 2], [360, 1, 2], [571, 2, 3], [1556, 2, 4], [2421, 3, 4]], [[23, 0, 1], [52, 0, 2], [517, 1, 2], [55, 2, 3]], [[608, 0, 1], [4445, 0, 2], [176, 1, 2]], [[244, 0, 1], [124, 0, 2], [1225, 1, 2]], [[246, 0, 1], [32313, 0, 2], [102, 1, 2]], [[131, 0, 1], [206, 0, 2], [2059, 1, 2], [55, 2, 3]], [[204, 0, 1], [1283, 0, 2], [55, 1, 2]], [[163, 0, 1], [4120, 0, 2], [395, 1, 2], [32, 1, 3], [226, 2, 3]]]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
    
        return output
    def pad_apart(self, id_seq, name, paras) :
        """
        id_seq :
        {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
         'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
         'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
         'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]}
        model inputs :
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

        paras['qu_l']  = 50,
        paras['pre_l'] = 20,
        paras['qu_i'] = [
            [[1,50]], 
            [[1,50], [3,60], [10,60]], 
            [[1,50], [3,90], [10,90], [30,90]], ]
        paras['pre_i'] = [
            [[1,20]], 
            [[1,20], [3,60], [10,60]], 
            [[1,20], [3,90], [10,90], [30,90]], ]
        "qu_sum1": 50,      "qu_sum2": 76,      "qu_sum3": 92,
        "pre_sum1": 20,     "pre_sum2": 46,     "pre_sum3": 62
        """
        # 下面这个长度就是除去最初pad那一位之后的所有核心位置~
        # print(name, (len(id_seq['seq'])-1), len(id_seq['uig']), len(id_seq['big']), len(id_seq['tig']))
        assert (len(id_seq['seq'])-1)==len(id_seq['uig'])==len(id_seq['big'])==len(id_seq['tig'])


        to_ret = {name:id_seq['seq']}

        """"""
        def get_min_id(x, l) :
            i_list = [t for t in l if t >= x]
            if len(i_list) == 0 :
                """   如果要看 check_report_infor 的话，这里一定要报错~   """
                return -1
            return i_list[0]
        iinfor = paras['%s_i'%(name)]
        i2k = {1:'uig', 2:'big', 3:'tig'}
        for i, infor_t in enumerate(iinfor) :
            # infor_t = [[1,50], [3,90], [10,90], [30,90]], ]
            ig_k = i2k[i+1]                     # 'tig'
            igs  = id_seq[ig_k]                 # [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
            len_list = [t[0] for t in infor_t]  # [1, 3, 10, 30]

            
            convert_name = '%s_convert%d'%(name, i+1)
            to_ret[convert_name] = [-1]*(len(igs)+1)

            temp_base = 0 # 当前的position迁移量~
            for infor_tt in infor_t :
                infor_l = infor_tt[0]

                name_t = '%s_%s_%d'%(name, ig_k, infor_l) # qu_tig_1 ~ qu_tig_30
                to_ret[name_t] = []

                for j, token_t in enumerate(igs) :
                    position = j+1 # 这是这个token在id_seq中对应的位置~
                    if get_min_id(len(token_t), len_list) == infor_l :
                        # # padding~ 统一申请空间之后，这里不用做padding~
                        # if len(token_t) < infor_l :
                        #     token_t += [[0]*len(token_t[0])]*(infor_l-len(token_t))

                        """
                        这是一个强行剪裁，当所属的这个to_ret[name_t]满了以后，将不再往里面放东西~
                        一定要将下面这行if注释掉再去看 check_report_infor 的结果~
                        （注释if这行，让下两句总是执行；注释pad整块，防止报错~；get_min_id函数中如果找不到，一定要报错~）
                        """
                        if len(to_ret[name_t]) == infor_tt[1]/infor_tt[0] :
                            break
                        to_ret[name_t].append(token_t)
                        to_ret[convert_name][position] = int(temp_base+len(to_ret[name_t])-1)
                temp_base += infor_tt[1] / infor_tt[0]

        """
        # print(to_ret)
        {'qu': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
         'qu_convert1': [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 
         'qu_uig_1': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
         'qu_convert2': [-1, 0, 50, 1, 2, 3, 51, 4, 5, 6, 7, 8, 52, 9, 10, 11], 
         'qu_big_1': [[[1, 2]], [[3, 5]], [[4, 6]], [[5, 6]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[13, 15]], [[14, 0]], [[15, 0]]], 
         'qu_big_3': [[[2, 3], [2, 4]], [[6, 7], [6, 8], [6, 9]], [[12, 13], [12, 14]]], 
         'qu_big_10': [], 
         'qu_convert3': [-1, 0, 50, 1, 2, 3, 80, 4, 5, 6, 7, 51, 81, 8, 9, 10], 
         'qu_tig_1': [[[0, 1, 2]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]], 
         'qu_tig_3': [[[1, 2, 3], [1, 2, 4]], [[8, 11, 12], [10, 11, 12]]], 
         'qu_tig_10': [[[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]]], 
         'qu_tig_30': []}

        padding之后：
        'qu_uig_1': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
        'qu_big_1': [[[1, 2]], [[3, 5]], [[4, 6]], [[5, 6]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[13, 15]], [[14, 0]], [[15, 0]]], 
        'qu_big_3': [[[2, 3], [2, 4], [0, 0]], [[6, 7], [6, 8], [6, 9]], [[12, 13], [12, 14], [0, 0]]], 
        'qu_big_10': [], 
        'qu_tig_1': [[[0, 1, 2]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]], 
        'qu_tig_3': [[[1, 2, 3], [1, 2, 4], [0, 0, 0]], [[8, 11, 12], [10, 11, 12], [0, 0, 0]]], 
        'qu_tig_10': [[[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], 
        'qu_tig_30': []}
        # for k in to_ret :
        #     print(k, np.shape(to_ret[k]))
        qu (16,)  qu_convert1 (16,)  qu_convert2 (16,)  qu_convert3 (16,)
        qu_uig_1 (15, 1, 1)
        qu_big_1 (12, 1, 2)  qu_big_3 (3, 3, 2)  qu_big_10 (0,)
        qu_tig_1 (11, 1, 3)  qu_tig_3 (2, 3, 3)  qu_tig_10 (2, 10, 3)  qu_tig_30 (0,)
        """

        """   check_report_infor   """
        # kas = [name]
        # kbs = ['len']
        # i2k = {1:'uig', 2:'big', 3:'tig'}
        # for i, iinfor_t in enumerate(iinfor) :
        #     for j in range(len(iinfor_t)) :
        #         kbs.append('%s_%d'%(i2k[i+1], j))
        # keys = ['%s_%s'%(ka, kb) for ka in kas for kb in kbs]
        # for k in keys :
        #     if not k in self.log_info :
        #         self.log_info[k] = []
        
        # self.log_info['%s_len'%(name)].append(len(to_ret[name]))
        # for i, iinfor_t in enumerate(iinfor) :
        #     for j in range(len(iinfor_t)) :
        #         key_ret = '%s_%s_%d'%(name, i2k[i+1], iinfor_t[j][0])
        #         key_log = '%s_%s_%d'%(name, i2k[i+1], j)
        #         self.log_info[key_log].append(len(to_ret[key_ret]))

        """   pad   """
        for i in [1,2,3] :
            key = '%s_convert%d'%(name, i)
            conver_t = to_ret[key]
            to_ret[key] = np.zeros([paras['%s_sum%d'%(name, i)], paras['%s_l'%(name)]])
            for i, t in enumerate(conver_t) :
                if t < 0 :
                    continue
                to_ret[key][t][i] = 1
            # print(to_ret[key])

        max_len = paras['%s_l'%(name)]
        to_ret[name] = np.pad(to_ret[name][:max_len], (0, max(0, max_len-len(to_ret[name]))), 'constant')
        i2k = {1:'uig', 2:'big', 3:'tig'}
        for i in [1,2,3] :
            # [[1, 20], [3, 90], [10, 90], [30, 90]]
            infort = iinfor[i-1]
            for infortt in infort :
                key = '%s_%s_%d'%(name, i2k[i], infortt[0])
                now_key = to_ret[key]
                to_ret[key] = np.zeros((infortt[1], i))
                
                for i0, t1 in enumerate(now_key) :
                    to_ret[key][i0*infortt[0]:i0*infortt[0]+len(t1)] = t1

                # 统一申请空间之后不用的版本~
                # if len(to_ret[key]) == 0 :
                #     to_ret[key] = np.zeros((infortt[1], i))
                # else :
                #     # (2,10,3) -> (20,3)
                #     t = [t2 for t1 in to_ret[key] for t2 in t1]
                #     # (20,3)   -> (90,3)
                #     to_ret[key] = np.pad(t, ((0, infortt[1]-len(t)), (0,0)), 'constant')
        # for k in to_ret :
        #     print(k, np.shape(to_ret[k]))
        """
        qu (50,)    pre (20,)

        qu_convert1 (50, 50)    qu_convert2 (76, 50)    qu_convert3 (92, 50)
        pre_convert1 (20, 20)   pre_convert2 (46, 20)   pre_convert3 (62, 20)

        qu_uig_1 (50, 1)
        qu_big_1 (50, 2)    qu_big_3 (60, 2)    qu_big_10 (60, 2)
        qu_tig_1 (50, 3)    qu_tig_3 (90, 3)    qu_tig_10 (90, 3)   qu_tig_30 (90, 3)

        pre_uig_1 (20, 1)
        pre_big_1 (20, 2)   pre_big_3 (60, 2)   pre_big_10 (60, 2)
        pre_tig_1 (20, 3)   pre_tig_3 (90, 3)   pre_tig_10 (90, 3)  pre_tig_30 (90, 3)


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

        return to_ret
    def pad_apart_check(self, id_seq, name, paras) :
        # pad_apart_count
        # 下面这个长度就是除去最初pad那一位之后的所有核心位置~
        # print(name, (len(id_seq['seq'])-1), len(id_seq['uig']), len(id_seq['big']), len(id_seq['tig']))
        assert (len(id_seq['seq'])-1)==len(id_seq['uig'])==len(id_seq['big'])==len(id_seq['tig'])


        to_ret = {name:id_seq['seq']}

        """"""
        def get_min_id(x, l) :
            i_list = [t for t in l if t >= x]
            if len(i_list) == 0 :
                """   如果要看 check_report_infor 的话，这里一定要报错~   """
                import hao123
                return -1
            return i_list[0]
        iinfor = paras['%s_i'%(name)]
        i2k = {1:'uig', 2:'big', 3:'tig'}
        for i, infor_t in enumerate(iinfor) :
            # infor_t = [[1,50], [3,90], [10,90], [30,90]], ]
            ig_k = i2k[i+1]                     # 'tig'
            igs  = id_seq[ig_k]                 # [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
            len_list = [t[0] for t in infor_t]  # [1, 3, 10, 30]

            
            convert_name = '%s_convert%d'%(name, i+1)
            to_ret[convert_name] = [-1]*(len(igs)+1)

            temp_base = 0 # 当前的position迁移量~
            for infor_tt in infor_t :
                infor_l = infor_tt[0]

                name_t = '%s_%s_%d'%(name, ig_k, infor_l) # qu_tig_1 ~ qu_tig_30
                to_ret[name_t] = []

                for j, token_t in enumerate(igs) :
                    position = j+1 # 这是这个token在id_seq中对应的位置~
                    if get_min_id(len(token_t), len_list) == infor_l :
                        # # padding~ 统一申请空间之后，这里不用做padding~
                        # if len(token_t) < infor_l :
                        #     token_t += [[0]*len(token_t[0])]*(infor_l-len(token_t))

                        """
                        这是一个强行剪裁，当所属的这个to_ret[name_t]满了以后，将不再往里面放东西~
                        一定要将下面这行if注释掉再去看 check_report_infor 的结果~
                        （注释if这行，让下两句总是执行；注释pad整块，防止报错~；get_min_id函数中如果找不到，一定要报错~）
                        """
                        # if len(to_ret[name_t]) == infor_tt[1]/infor_tt[0] :
                        #     break
                        to_ret[name_t].append(token_t)
                        to_ret[convert_name][position] = int(temp_base+len(to_ret[name_t])-1)
                temp_base += infor_tt[1] / infor_tt[0]

        

        """   check_report_infor   """
        kas = [name]
        kbs = ['len']
        i2k = {1:'uig', 2:'big', 3:'tig'}
        for i, iinfor_t in enumerate(iinfor) :
            for j in range(len(iinfor_t)) :
                kbs.append('%s_%d'%(i2k[i+1], j))
        keys = ['%s_%s'%(ka, kb) for ka in kas for kb in kbs]
        for k in keys :
            if not k in self.log_info :
                self.log_info[k] = []
        
        self.log_info['%s_len'%(name)].append(len(to_ret[name]))
        for i, iinfor_t in enumerate(iinfor) :
            for j in range(len(iinfor_t)) :
                key_ret = '%s_%s_%d'%(name, i2k[i+1], iinfor_t[j][0])
                key_log = '%s_%s_%d'%(name, i2k[i+1], j)
                self.log_info[key_log].append(len(to_ret[key_ret]))

        

        return to_ret
    def padding_nlpcckbqa_mws(self, xs) :
        """
        输入的xs中的每个x的示例~
        {'input': 
        {'qu': 
          {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
           'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
           'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
          }, 
         'pre': 
          {'seq': [0, 213, 71, 9, 158, 1170], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]]], 
           'big': [[[1, 4]], [[2, 5]], [[3, 0]], [[4, 5]], [[5, 0]]], 
           'tig': [[[0, 1, 4]], [[0, 2, 5]], [[0, 3, 0]], [[1, 4, 5]], [[2, 5, 0], [4, 5, 0]]]
          }
        }, 
         'output': {'label': 1}}
        """

        sys.path.append('model')
        from model_nlpcc_kbre_cws import get_para
        paras = get_para()
        """
        paras['qu_l']  = 50,
        paras['pre_l'] = 20,
        paras['qu_i'] = [
            [[1,50]], 
            [[1,50], [3,60], [10,60]], 
            [[1,50], [3,90], [10,90], [30,90]], ]
        paras['pre_i'] = [
            [[1,20]], 
            [[1,20], [3,60], [10,60]], 
            [[1,20], [3,90], [10,90], [30,90]], ]
        "qu_sum1": 50,      "qu_sum2": 76,      "qu_sum3": 92,
        "pre_sum1": 20,     "pre_sum2": 46,     "pre_sum3": 62
        """

        inputs = []
        for x in xs :
            to_input =  self.pad_apart(x['input']['qu'], name='qu', paras=paras)
            to_input.update(self.pad_apart(x['input']['pre'], name='pre', paras=paras))
            inputs.append(to_input)

        output_keys = xs[0]['output'].keys()
        outputs = {key:np.array([x['output'][key] for x in xs])
            for key in output_keys}

        input_keys = inputs[0].keys()
        inputs = {key:np.array([x[key] for x in inputs])
            for key in input_keys}
        

        return inputs, outputs

    """对nlpcckbqa gcn 的预处理"""
    def deal_nlpcckbqa_gcn_base(self, x, len_qu=50, len_pre=20) :
        return self.deal_nlpcckbqa_mws_base(x, len_qu=len_qu, len_pre=len_pre)
    def deal_dgcn_seq(self, id_seq, lent, name, max_pad=5) :
        """
        {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
         'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
         'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
         'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
        }
        """
        reproduce = lambda x : sorted(list(set(x)))
        
        seq = id_seq['seq'][:lent]
        seq = seq + [0]*(lent-len(seq))

        tig = id_seq['tig'][:lent-1]
        mid_ids = [t[0][1] for t in tig] + [0]*(lent-len(tig))
        head_ids = [reproduce([tt[0] for tt in t]) for t in tig]
        tail_ids = [reproduce([tt[2] for tt in t]) for t in tig]

        head_ids = pad_sequences(head_ids, maxlen=max_pad, padding='post', truncating='post')
        tail_ids = pad_sequences(tail_ids, maxlen=max_pad, padding='post', truncating='post')

        head_ids = np.pad(head_ids, ((0, lent-len(head_ids)), (0,0)), 'constant')
        tail_ids = np.pad(tail_ids, ((0, lent-len(tail_ids)), (0,0)), 'constant')
        
        to_ret = {
            name             : seq, 
            '%s_mid'%(name)  : mid_ids,
            '%s_head'%(name) : head_ids,
            '%s_tail'%(name) : tail_ids,
        }
        return to_ret
    def padding_nlpcckbqa_gcn(self, xs, len_qu=50, len_pre=20) :
        """
        输入的xs中的每个x的示例~
        {'input': 
        {'qu': 
          {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
           'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
           'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
          }, 
         'pre': 
          {'seq': [0, 213, 71, 9, 158, 1170], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]]], 
           'big': [[[1, 4]], [[2, 5]], [[3, 0]], [[4, 5]], [[5, 0]]], 
           'tig': [[[0, 1, 4]], [[0, 2, 5]], [[0, 3, 0]], [[1, 4, 5]], [[2, 5, 0], [4, 5, 0]]]
          }
        }, 
         'output': {'label': 1}}
        """

        inputs = []
        for x in xs :
            to_input    =   self.deal_dgcn_seq(x['input']['qu'], len_qu, name='qu')
            to_input.update(self.deal_dgcn_seq(x['input']['pre'], len_pre, name='pre'))
            inputs.append(to_input)

        output_keys = xs[0]['output'].keys()
        outputs = {key:np.array([x['output'][key] for x in xs])
            for key in output_keys}

        input_keys = inputs[0].keys()
        inputs = {key:np.array([x[key] for x in inputs])
            for key in input_keys}

        return inputs, outputs

    """对nlpccdbqa word 的预处理"""
    def deal_nlpccirqa_word_base(self, x) :
        # none_feature_words=set(['<NW_uni>', '<NW_bi>', '<NW_tri>'])
        none_feature_words=set()
        get_features = lambda group, seq:[1. if all([t in group, not t in none_feature_words]) else 0 for t in seq]
        output = {}
        get_ids = lambda x : [self.w2v.getWordID(t) for t in x]

        output['sent_features'] = [get_features(set(x['qu']), t) for t in x['sents']]
        output['qu_features']   = [get_features(set(t), x['qu']) for t in x['sents']]

        output['qu_ids'] = get_ids(x['qu'])
        output['sent_ids'] = [get_ids(t) for t in x['sents']]
        output['sent_labels'] = x['labels']
        """
        print (output['qu_ids'])
        print (output['pre_ids'])
        print (output['pre_labels'])
        [155840, 198, 690, 1, 31, 3, 1457]
        [[3879], [40353], [29904], [70, 36], [177], [508, 54], [299, 175], [1782], [8], [31], [388, 1555], [51, 54], [4444], [123], [245, 101], [205, 54], [1282], [4119, 225]]
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        """
    
        return output
    def deal_nlpccirqa_train(self, x_base, num=10) :
        pre_len = len(x_base['sent_ids'])

        self.__get_seed(random.seed)
        index = list(range(pre_len))
        random.shuffle(index)

        true_id = [i for i in range(pre_len) if x_base['sent_labels'][i] == 1][0]
        if not true_id in index[:num] :
            index = [true_id]+index
        index = index[:num]

        output = [{'input':{'qu':x_base['qu_ids'], 
                   'sent':x_base['sent_ids'][idt],
                   'qu_feature':x_base['qu_features'][idt],
                   'sent_feature':x_base['sent_features'][idt],}, 
                   'output':{'label':x_base['sent_labels'][idt]}}
            for idt in index]

        return output
    def deal_nlpccirqa_test(self, x_base) :
        output = [{'input':{'qu':x_base['qu_ids'], 
                   'sent':x_base['sent_ids'][idt], 
                   'qu_feature':x_base['qu_features'][idt],
                   'sent_feature':x_base['sent_features'][idt],}, 
                   'output':{'label':x_base['sent_labels'][idt]}}
            for idt in range(len(x_base['sent_ids']))]
        return output
    def padding_nlpccirqa_word(self, xs, len_qu=60, len_pre=120) :
        """   返回字典构建   """
        outputs = [t['output'] for t in xs]
        inputs  = [t['input']  for t in xs]
        output_keys = outputs[0].keys()
        input_keys  = inputs[0].keys()
        outputs = {key:np.array([t[key] for t in outputs])
            for key in output_keys}
        inputs  = {key:np.array([t[key] for t in inputs])
            for key in input_keys}

        """   padding   """
        head_words = np.array([0, 0, 0, 0])
        for key in ['qu', 'sent', 'qu_feature', 'sent_feature'] :
            inputs[key] = list(inputs[key])
            matrix = inputs[key]
            for i, line in enumerate(matrix) :
                line = np.array(line)
                t= np.concatenate((head_words, line))
                matrix[i] = t 

        inputs['qu']  = pad_sequences(inputs['qu'], maxlen=len_qu, padding='post', truncating='post')
        inputs['sent'] = pad_sequences(inputs['sent'], maxlen=len_pre, padding='post', truncating='post')
        inputs['qu_feature']  = pad_sequences(inputs['qu_feature'], maxlen=len_qu, padding='post', truncating='post')
        inputs['sent_feature'] = pad_sequences(inputs['sent_feature'], maxlen=len_pre, padding='post', truncating='post')

        return inputs, outputs

    """对nlpccdbqa mws 的预处理"""
    def deal_nlpccirqa_mws_base(self, x, len_qu=80, len_pre=160) :
        output = {}
        """   获取 word-overlap 特征，目前可能还有些问题~   """
        # none_feature_words=set(['<NW_uni>', '<NW_bi>', '<NW_tri>'])
        none_feature_words=set()
        get_features = lambda group, seq:[1. if all([t[0] in group, not t[0] in none_feature_words]) else 0. for t in seq]

        output['sent_features'] = [[0.]+get_features(set([tt[0] for tt in x['qu']]), t) for t in x['sents']]
        output['qu_features']   = [[0.]+get_features(set([tt[0] for tt in t]), x['qu']) for t in x['sents']]
        output['sent_features'] = pad_sequences(output['sent_features'], maxlen=len_pre, padding='post', truncating='post')
        output['qu_features']   = pad_sequences(output['qu_features'], maxlen=len_qu, padding='post', truncating='post')

        """   转为ID   """
        get_ids = lambda x : [[self.w2v.getWordID(t[0]), t[1], t[2]] for t in x]

        output['qu_ids'] = get_ids(x['qu'])
        output['sent_ids'] = [get_ids(t) for t in x['sents']]

        qu_deals   = self.deal_lattice_base(output['qu_ids'], max_len=len_qu)
        pres_deals = [self.deal_lattice_base(t, max_len=len_pre) for t in output['sent_ids']]
        output['qu_ids'] = qu_deals
        output['sent_ids'] = pres_deals

        output['sent_labels'] = x['labels']

        """
        print (output['qu_ids'])
        print (output['pre_ids'])
        print (output['pre_labels'])
        [[155841, 0, 1], [199, 1, 2], [102, 2, 3], [691, 2, 4], [204, 3, 4], [2, 4, 5], [213, 5, 6], [71, 5, 7], [9, 5, 8], [158, 6, 7], [1170, 7, 8], [4, 8, 9], [859, 9, 10], [5506, 9, 11], [49, 10, 11]]
        [[[1226, 0, 1], [3880, 0, 2], [4223, 1, 2]], [[4885, 0, 1], [21775, 0, 2], [40354, 0, 4], [4276, 1, 2], [62040, 1, 3], [3856, 2, 3], [45976, 2, 4], [3511, 3, 4]], [[60, 0, 1], [29905, 0, 3], [2053, 1, 2], [958, 1, 3], [1050, 2, 3]], [[213, 0, 1], [71, 0, 2], [158, 1, 2], [183, 2, 3], [37, 2, 4], [486, 3, 4]], [[1298, 0, 1], [178, 0, 2], [18368, 1, 2]], [[4962, 0, 1], [509, 0, 2], [204, 1, 2], [1283, 1, 3], [55, 2, 3]], [[300, 0, 1], [176, 1, 2]], [[1225, 0, 1], [1783, 0, 2], [55, 1, 2]], [[213, 0, 1], [71, 0, 2], [9, 0, 3], [158, 1, 2], [1170, 2, 3]], [[395, 0, 1], [32, 0, 2], [226, 1, 2]], [[12543, 0, 1], [389, 0, 2], [360, 1, 2], [571, 2, 3], [1556, 2, 4], [2421, 3, 4]], [[23, 0, 1], [52, 0, 2], [517, 1, 2], [55, 2, 3]], [[608, 0, 1], [4445, 0, 2], [176, 1, 2]], [[244, 0, 1], [124, 0, 2], [1225, 1, 2]], [[246, 0, 1], [32313, 0, 2], [102, 1, 2]], [[131, 0, 1], [206, 0, 2], [2059, 1, 2], [55, 2, 3]], [[204, 0, 1], [1283, 0, 2], [55, 1, 2]], [[163, 0, 1], [4120, 0, 2], [395, 1, 2], [32, 1, 3], [226, 2, 3]]]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
    
        return output
    def padding_nlpccirqa_mws(self, xs) :
        """
        输入的xs中的每个x的示例~
        {'input': 
        {'qu': 
          {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
           'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
           'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
          }, 
         'pre': 
          {'seq': [0, 213, 71, 9, 158, 1170], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]]], 
           'big': [[[1, 4]], [[2, 5]], [[3, 0]], [[4, 5]], [[5, 0]]], 
           'tig': [[[0, 1, 4]], [[0, 2, 5]], [[0, 3, 0]], [[1, 4, 5]], [[2, 5, 0], [4, 5, 0]]]
          }
        }, 
         'output': {'label': 1}}
        """
        
        if not self.pad_info is None :
            paras = self.pad_info
        else :
            sys.path.append('model')
            from model_nlpcc_irqa_cws import get_para
            paras = get_para()
            self.pad_info = paras

        """
        paras['qu_l']  = 50,
        paras['pre_l'] = 20,
        paras['qu_i'] = [
            [[1,50]], 
            [[1,50], [3,60], [10,60]], 
            [[1,50], [3,90], [10,90], [30,90]], ]
        paras['pre_i'] = [
            [[1,20]], 
            [[1,20], [3,60], [10,60]], 
            [[1,20], [3,90], [10,90], [30,90]], ]
        "qu_sum1": 50,      "qu_sum2": 76,      "qu_sum3": 92,
        "pre_sum1": 20,     "pre_sum2": 46,     "pre_sum3": 62
        """

        inputs = []
        for x in xs :
            to_input =  self.pad_apart(x['input']['qu'], name='qu', paras=paras)
            to_input.update(self.pad_apart(x['input']['sent'], name='sent', paras=paras))

            # pad feature， 最初的0是因为id_seq中最开始有一个pad~ # 修改@1，挪到了base
            # to_input['qu_feature']   = pad_sequences([[0]+x['input']['qu_feature']], maxlen=paras['qu_l'], padding='post', truncating='post')[0]
            # to_input['sent_feature'] = pad_sequences([[0]+x['input']['sent_feature']], maxlen=paras['sent_l'], padding='post', truncating='post')[0]
            to_input['qu_feature']   = x['input']['qu_feature']
            to_input['sent_feature'] = x['input']['sent_feature']
            inputs.append(to_input)

        output_keys = xs[0]['output'].keys()
        outputs = {key:np.array([x['output'][key] for x in xs])
            for key in output_keys}

        input_keys = inputs[0].keys()
        # inputs = {key:np.array([x[key] for x in inputs])
        #     for key in input_keys}
        inputs = {key:np.concatenate([np.expand_dims(x[key], 0) for x in inputs], axis=0)
            for key in input_keys}


        return inputs, outputs

    """对nlpccdbqa gcn 的预处理"""
    def deal_nlpccirqa_gcn_base(self, x, len_qu=80, len_pre=160) :
        return self.deal_nlpccirqa_mws_base(x, len_qu=len_qu, len_pre=len_pre)
    def padding_nlpccirqa_gcn(self, xs, len_qu=80, len_pre=160) :
        """
        输入的xs中的每个x的示例~
        {'input': 
        {'qu': 
          {'seq': [0, 155841, 199, 102, 691, 204, 2, 213, 71, 9, 158, 1170, 4, 859, 5506, 49], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]], [[9]], [[10]], [[11]], [[12]], [[13]], [[14]], [[15]]], 
           'big': [[[1, 2]], [[2, 3], [2, 4]], [[3, 5]], [[4, 6]], [[5, 6]], [[6, 7], [6, 8], [6, 9]], [[7, 10]], [[8, 11]], [[9, 12]], [[10, 11]], [[11, 12]], [[12, 13], [12, 14]], [[13, 15]], [[14, 0]], [[15, 0]]], 
           'tig': [[[0, 1, 2]], [[1, 2, 3], [1, 2, 4]], [[2, 3, 5]], [[2, 4, 6]], [[3, 5, 6]], [[4, 6, 7], [5, 6, 7], [4, 6, 8], [5, 6, 8], [4, 6, 9], [5, 6, 9]], [[6, 7, 10]], [[6, 8, 11]], [[6, 9, 12]], [[7, 10, 11]], [[8, 11, 12], [10, 11, 12]], [[9, 12, 13], [11, 12, 13], [9, 12, 14], [11, 12, 14]], [[12, 13, 15]], [[12, 14, 0]], [[13, 15, 0]]]
          }, 
         'pre': 
          {'seq': [0, 213, 71, 9, 158, 1170], 
           'uig': [[[1]], [[2]], [[3]], [[4]], [[5]]], 
           'big': [[[1, 4]], [[2, 5]], [[3, 0]], [[4, 5]], [[5, 0]]], 
           'tig': [[[0, 1, 4]], [[0, 2, 5]], [[0, 3, 0]], [[1, 4, 5]], [[2, 5, 0], [4, 5, 0]]]
          }
        }, 
         'output': {'label': 1}}
        """

        inputs = []
        for x in xs :
            to_input    =   self.deal_dgcn_seq(x['input']['qu'], len_qu, name='qu')
            to_input.update(self.deal_dgcn_seq(x['input']['sent'], len_pre, name='sent'))
            to_input['qu_feature']   = x['input']['qu_feature']
            to_input['sent_feature'] = x['input']['sent_feature']
            inputs.append(to_input)

        output_keys = xs[0]['output'].keys()
        outputs = {key:np.array([x['output'][key] for x in xs])
            for key in output_keys}

        input_keys = inputs[0].keys()
        inputs = {key:np.array([x[key] for x in inputs])
            for key in input_keys}

        return inputs, outputs

if __name__ == '__main__':
    pass
    """单元测试"""

    preprocess = CPreprocess(w2v_path = '../data_vectors/vectorsw300l20.all')

    # data = json.load(codecs.open('../data/test_re_9413_segmws.json', 'r', 'utf-8'))
    # data = json.load(codecs.open('../data/test_re_9413_segword.json', 'r', 'utf-8'))
    
    dta0 = [preprocess.deal_nlpccdbqa_nword_base(data[i]) for i in range(2)]
    t = preprocess.deal_nlpccdbqa_nword_train(dta0[0])
    ti, to = preprocess.padding_nlpccdbqa_nword_word(t)
    print(t)
    for k in ti :
        print(k, np.shape(ti[k]))
    for k in to :
        print(k, np.shape(to[k]))
    print(ti)
    print(to)
    exit()

    
