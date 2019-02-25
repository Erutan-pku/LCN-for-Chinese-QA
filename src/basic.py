#coding=utf-8
#-*- coding: UTF-8 -*-
import sys

"""python3"""

import os
import json
import time
import jieba
import codecs
import random
import argparse
import numpy as np
from tqdm import tqdm

class W2V :
    def __init__(self, wordVecFile, split=' ', verbose=False) :
        input_file = codecs.open(wordVecFile, encoding='utf-8')

        pad_word = '<padding>'
        self.w2i = {}
        self.i2v = {}
        self.wset = set()

        start =True
        if verbose :
            input_file = tqdm(input_file)
        for line in input_file:
            if start :
                start = False
                p0, p1 = map(int, line.split(' ')) # 155837 300
                self.w2i[pad_word] = 0
                self.i2v[0] = np.zeros(p1)
                self.wset.add(pad_word)
                continue

            line = line.strip().split(split)
            word = line[0]
            vec = np.array([float(t) for t in line[1:]])

            self.w2i[word] = len(self.w2i)
            self.i2v[self.w2i[word]] = vec
        self.wset = set(self.w2i.keys())
        self.size = len(self.wset)

        vec_lens = set([len(self.i2v[k]) for k in self.i2v])
        assert len(vec_lens) == 1
        self.vec_len = list(vec_lens)[0]

        uni_word = [self.w2i[t] for t in self.wset if len(t) == 1]
        bi_word = [self.w2i[t] for t in self.wset if len(t) == 2]
        tri_word = [self.w2i[t] for t in self.wset if len(t) == 3]
        oi_word = [self.w2i[t] for t in self.wset if len(t) > 3]
        groups = [uni_word, bi_word, tri_word, oi_word]
        average_vectors = [sum([self.i2v[t] for t in group]) / len(group) for group in groups]
        names = ['<NW_uni>', '<NW_bi>', '<NW_tri>', '<NW_oi>']
        for n, v in zip(names, average_vectors) :
            self.addWord(n, v)

    def __len__(self) :
        return len(self.wset)

    def getVecLen(self) :
        return self.vec_len

    def addWord(self, word, npArray) :
        assert not word in self.wset
        self.wset.add(word)
        self.size = len(self.wset)
        
        self.w2i[word] = len(self.w2i)
        self.i2v[self.w2i[word]] = npArray

    def getWordID(self, word) :
        if word in self.wset :
            return self.w2i[word]
        if len(word) == 1 :
            return self.w2i['<NW_uni>']
        if len(word) == 2 :
            return self.w2i['<NW_bi>']
        if len(word) == 3 :
            return self.w2i['<NW_tri>']
        return self.w2i['<NW_oi>']

    def getWordVector(self, word) :
        if not word in self.wset :
            return None

        v = np.array(self.i2v[self.getWordID(word)])
        return v

    def getMatrix(self) :
        ret = []
        for i in range(self.size) :
            ret.append(self.i2v[i])
        return np.array(ret)


class CWS :
    def __init__ (self, wordBank=None) :
        if not wordBank is None :
            if type(wordBank) is str :
                input_file = codecs.open(wordBank, 'r', 'utf-8')
                self.wordBank = set([t.strip() for t in input_file])
            else :
                self.wordBank = set(wordBank)
    def getWordSet (self) :
        assert not self.wordBank is None
        return self.wordBank
    
    """simple rule tools"""
    def MM(self, sentence, max_n=20) :
        assert not self.wordBank is None

        start_loc = 0
        wordList = []

        while start_loc < len(sentence) :
            mark_have = False
            for end_loc in range(min(start_loc + max_n, len(sentence)), start_loc + 1, -1) :
                word_t = sentence[start_loc:end_loc]
                if word_t in self.wordBank :
                    wordList.append(word_t)
                    start_loc = end_loc
                    mark_have = True
                    break
            if not mark_have :
                wordList.append(sentence[start_loc])
                start_loc += 1
        wordList = [word for word in wordList if not word == ' ']
        return wordList
    def RMM(self, sentence, max_n = 20) :
        assert not self.wordBank is None
        
        start_loc = 0
        wordList = []
        sentence = sentence[::-1]

        while start_loc < len(sentence) :
            mark_have = False
            for end_loc in range(min(start_loc + max_n, len(sentence)), start_loc + 1, -1) :
                word_t = sentence[start_loc:end_loc]
                word_t = word_t[::-1]
                if word_t in self.wordBank :
                    wordList.append(word_t)
                    start_loc = end_loc
                    mark_have = True
                    break
            if not mark_have :
                wordList.append(sentence[start_loc])
                start_loc += 1
        wordList = [word for word in wordList if not word == ' ']
        return wordList[::-1]
    def MWS(self, sentence, index=False, max_len=20) :
        assert not self.wordBank is None
        wordList = []
        for i in range(len(sentence)) :
            for j in range(i+1, len(sentence) + 1) :
                if j - i > max_len :
                    break
                word_t = sentence[i:j]
                if word_t in self.wordBank :
                    if index :
                        wordList.append([word_t, i, j]);
                    else :
                        wordList.append(word_t);
        return wordList
    
    """simple jieba"""
    def jieba_cws(self, sentence) :
        return ' '.join(jieba.cut(sentence)).split(' ')
    def jieba_postOnly(self, sentence) :
        assert not type(sentence) is list
        return [w for w, s in jieba.posseg.cut(sentence)]
    def jieba_PosWod(self, sentence) :
        assert not type(sentence) is list
        return [[w, s]for w, s in jieba.posseg.cut(sentence)]

    """一些复杂的，或者是组合的方法"""
    # 在jieba分词的基础上，再用正向匹配法分个词
    def jieba_cws2(self, sentence) :
        assert not self.wordBank is None
        result = []
        words = ' '.join(jieba.cut(sentence)).split(' ')
        for word in words :
            result += self.MM(word)
        return result

def check_unk_seg(input_path, word_bank_path) :
    codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')
    json_load = lambda x : json.load(codecs.open(x, 'r', 'utf-8'))
    
    print(word_bank_path)
    word_bank = codecs_in(word_bank_path).readlines()
    word_bank = set([t.strip() for t in word_bank])

    can = 'sents'
    datas = json_load(input_path)
    all_words = 0
    in_words = 0
    for dt in datas :
        all_words += len(dt['qu'])
        in_words += len([t for t in dt['qu'] if t[0] in word_bank])
        for sent in dt[can] :
            all_words += len(sent)
            in_words += len([t for t in sent if t[0] in word_bank])
    print(100.*in_words/all_words, in_words, all_words)

def norm_embed(input_path, output_path) :
    codecs_in = lambda x : codecs.open(x, 'r', 'utf-8')

    input_file = codecs_in(input_path)
    head_line = None
    for line in tqdm(input_file) :
        if head_line is None :
            head_line = line
            continue

        ls = line.strip().split(' ')



# check_unk_seg('../train_dbqa_8768_seg_jieba_mws.small.json', '../data_vectors/WordBank_v')
# check_unk_seg('../train_dbqa_8768_segword_mws.small.json', '../data_vectors/WordBank_v')





