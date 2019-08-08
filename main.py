# -*- coding: utf-8 -*-
"""
 @Time    : 2019-08-08 18:34
 @FileName: main.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import re
import torch
import sentencepiece as spm

from model import BertModel

sp = spm.SentencePieceProcessor()
sp.load('resource/sentencepiece.unigram.35000.model')
vocab_size = sp.get_piece_size()

n_embedding = 512
n_layer = 8

model = BertModel(vocab_size, n_embedding, n_layer)
model.eval()
model.load_state_dict(torch.load('resource/model.{}.{}.th'.format(n_embedding, n_layer),
                                 map_location='cpu'))

# you should enable cuda if it is available
# model.cuda()

# if you are using a GPU that has tensor cores (nvidia volta, Turing architecture), you can enable half precision
# inference and training, we recommend to use the nvidia official apex to make everything as clean as possible from
# apex import amp [model] = amp.initialize([model], opt_level="O2")
device = model.embedding.weight.data.device


def clean_text(txt):
    txt = txt.lower()
    txt = re.sub('\s*', '', txt)
    return txt


def get_sequence_ids(text):
    return sp.EncodeAsIds(clean_text(text))


def get_sequence_embedding(text):
    ids = [get_sequence_ids(text)]
    ids = torch.LongTensor(ids).to(device)
    with torch.no_grad():
        hidden = model.inference(ids)
    return hidden


def get_pair_embedding(s1, s2):
    """
    during training, we use sentence pairs to train bert,
     so it could also be utilized for inference
    :param s1: sentence 1, usually the question, premise
    :param s2: sentence 1, such as the document, hypothesis
    :return: embedding
    """
    s1_ids = get_sequence_ids(s1)
    s2_ids = get_sequence_ids(s2)
    ids = [s1_ids + [vocab_size] + s2_ids]
    ids = torch.LongTensor(ids).to(device)
    with torch.no_grad():
        hidden = model.inference(ids)
    return hidden


if __name__ == '__main__':
    sentence = '我们一起学猫叫，一起喵喵喵喵'
    embedding = get_sequence_embedding(sentence)
    print(embedding.size())
