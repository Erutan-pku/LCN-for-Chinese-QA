#coding=utf-8
#-*- coding: UTF-8 -*-
import sys
import numpy as np

"""python3"""


def recover_kbqa(data, predication) :
    pre_key = 'pres'
    if not pre_key in data[0] :
        pre_key = 'pres_0'
    t = int(sum([len(t[pre_key]) for t in data]))
    predication = list(predication[:t])
    # assert len(predication) == sum([len(t['pres']) for t in data])
    # predication = predication[:sum([len(t['sents']) for t in data])]
    
    index = -1
    scores = {'len':len(data), 'len_pre':len(predication)}
    other_keys = ['r@1', 'r@2', 'r@3', 'r@5', 'r@10', 'MRR']
    for k in other_keys :
        scores[k] = 0.

    for dt in data :
        dt['scores'] = []
        for pre in dt[pre_key] :
            index+=1
            dt['scores'].append(predication[index][0])
        zip_scores = zip(dt['scores'], dt['labels'])
        zip_scores = sorted(zip_scores, key=lambda x:x[0], reverse=True)
        head_id = [i for i, t in enumerate(zip_scores) if t[1]==1][0]
        
        """   更新评价指标   """
        if head_id < 10 :
            scores['r@10'] += 1
        if head_id < 5 :
            scores['r@5'] += 1
        if head_id < 3 :
            scores['r@3'] += 1
        if head_id < 2 :
            scores['r@2'] += 1
        if head_id < 1 :
            scores['r@1'] += 1
        scores['MRR'] += 1./(head_id+1)

    for k in other_keys :
        scores[k] /= scores['len']

    return data, scores

def recover_dbqa(data, predication) :
    pre_key = 'sents'
    if not pre_key in data[0] :
        pre_key = 'sents_0'
    assert len(predication) == sum([len(t[pre_key]) for t in data])
    
    index = -1
    scores = {'len':len(data), 'len_sent':len(predication)}
    other_keys = ['r@1', 'r@2', 'r@3', 'r@5', 'MRR', 'MAP']
    for k in other_keys :
        scores[k] = 0.

    for dt in data :
        dt['scores'] = []
        for pre in dt[pre_key] :
            index+=1
            dt['scores'].append(predication[index][0])
        zip_scores = zip(dt['scores'], dt['labels'])
        zip_scores = sorted(zip_scores, key=lambda x:x[0], reverse=True)
        head_ids = [i for i, t in enumerate(zip_scores) if t[1]==1]
        head_id = head_ids[0]
        
        """   更新评价指标   """
        if head_id < 5 :
            scores['r@5'] += 1
        if head_id < 3 :
            scores['r@3'] += 1
        if head_id < 2 :
            scores['r@2'] += 1
        if head_id < 1 :
            scores['r@1'] += 1
        scores['MRR'] += 1./(head_id+1)
        scores['MAP'] += sum([(i+1.)/(t+1.) for i, t in enumerate(head_ids)]) / len(head_ids)

    for k in other_keys :
        scores[k] /= scores['len']

    return data, scores



