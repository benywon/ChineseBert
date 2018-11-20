# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/20 上午10:38
 @FileName: interface.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import pickle
import re
import jieba_fast as jieba
import torch
import numpy as np
from model import MultiHead


def get_model(filename):
    with open(filename, 'rb') as f:
        model = torch.load(f, map_location=torch.device('cpu'))
    return model


def clean(txt):
    txt = txt.lower()
    return re.sub('\s*', '', txt)


def padding(sequence, pads=0, max_len=None, dtype='int32', return_matrix_for_size=False):
    # we should judge the rank
    v_length = [len(x) for x in sequence]  # every sequence length
    seq_max_len = max(v_length)
    if (max_len is None) or (max_len > seq_max_len):
        max_len = seq_max_len
    v_length = list(map(lambda z: z if z <= max_len else max_len, v_length))
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
    if return_matrix_for_size:
        v_matrix = np.asanyarray([map(lambda item: 1 if item < line else 0, range(max_len)) for line in v_length],
                                 dtype=dtype)
        return x, v_matrix
    return x, np.asarray(v_length, dtype='int32')


class DQUAM(object):
    def __init__(self,
                 n_layer=16,
                 n_hidden=1024):
        self.model = MultiHead(52777, n_hidden, n_hidden, 16)
        self.model.cpu()
        self.model.eval()
        self.model.load_state_dict(get_model('model.{}.{}.th'.format(n_hidden, n_layer)))
        with open('word2id.obj', 'rb') as f:
            self.word2id = pickle.load(f)
        self.max_length = max([len(x) for x in self.word2id])
        self.vocab_size = len(self.word2id)

    def get_string_id_fmm(self, line):
        output = []
        lst = list(line)
        length = len(lst)
        start = 0
        while start < length:
            end = start + self.max_length
            end = end if end < length else length
            for i in range(end, start, -1):
                word = ''.join(lst[start:i])
                if word in self.word2id:
                    output.append(self.word2id[word])
                    break
                elif i == start + 1:
                    output.append(1)
            start = i
        return output

    def get_string_id(self, line):
        lst = jieba.lcut(clean(line))
        output = []
        for word in lst:
            if word in self.word2id:
                output.append(self.word2id[word])
            else:
                output.extend(self.get_string_id_fmm(word))
        return output

    def get_embedding(self, question, answer):
        batch = [self.get_string_id(x) + [self.vocab_size] + self.get_string_id(y) for x, y in zip(question, answer)]
        batch, _ = padding(batch)
        inputs = torch.LongTensor(batch)
        output = self.model(inputs)
        return output


if __name__ == '__main__':
    question = ['谁是杀死朱元璋的凶手？']
    answer = ['我们都没有遇到过您说的这个情况，如果有问题可以请教这边的咨询助理']
    model = DQUAM()
    print(model.get_embedding(question, answer).size())
