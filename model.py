# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 下午7:28
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    import torch.nn.LayerNorm as LayerNorm

try:
    import F.gelu as gelu
except ImportError:
    def gelu(x):
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiHeadBlock(nn.Module):
    def __init__(self, n_input, n_head=6):
        super().__init__()
        self.combined_projection = nn.Linear(n_input, 2 * (n_input // n_head) * n_head + (n_input // 2) * n_head)
        self.output_projection = nn.Linear((n_input // 2) * n_head, n_input)
        self.mlp = nn.Sequential(nn.Linear(n_input, n_input),
                                 nn.SELU(inplace=True),
                                 nn.Linear(n_input, n_input),
                                 )
        nn.init.xavier_normal_(self.combined_projection.weight, gain=0.1)
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.1)
        # nn.init.xavier_normal_(self.inter.weight, gain=0.1)

        self._scale = (n_input // n_head) ** 0.5
        self.att_dim = (n_input // n_head) * n_head
        self.num_heads = n_head

        self.dropout = nn.Dropout(p=0.1)
        # self.ln = apex.normalization.FusedLayerNorm(n_input)
        self.ln = LayerNorm(n_input)

    def forward(self, representations):
        batch_size, timesteps, _ = representations.size()
        combined_projection = F.leaky_relu(self.combined_projection(representations), inplace=True)
        queries, keys, *values = combined_projection.split(self.att_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        values_per_head = values.view(batch_size, timesteps, self.num_heads, values.size(-1) // self.num_heads)
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * self.num_heads, timesteps,
                                               values.size(-1) // self.num_heads)

        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        keys_per_head = keys.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * self.num_heads, timesteps, self.att_dim // self.num_heads)

        similarities = queries_per_head.bmm(keys_per_head.transpose(2, 1)) / self._scale

        similarities = F.softmax(similarities, 2)

        outputs = similarities.bmm(values_per_head)

        outputs = outputs.view(batch_size, self.num_heads, timesteps, values.size(-1) // self.num_heads)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, values.size(-1))

        inter = self.ln(representations + self.dropout(self.output_projection(outputs)))

        hidden = gelu(self.mlp(inter))

        return inter + hidden


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6, mm=True):
        super().__init__()
        self.n_head = n_head
        self.att = nn.ModuleList()
        for l in range(n_layer):
            en = MultiHeadBlock(n_hidden, n_head=n_head)
            ln = LayerNorm(n_hidden)
            # ln = apex.normalization.FusedLayerNorm(n_hidden)
            self.att.append(nn.Sequential(en, ln))

    def forward(self, representations):
        for one in self.att:
            representations = one(representations)
        return representations


class BertModel(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_layer):
        super().__init__()
        vocabulary_size = (2 + vocab_size // 8) * 8
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding)
        self.n_embedding = n_embedding
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_embedding // 2, bidirectional=True, num_layers=1,
                               batch_first=True)
        self.attention = SelfAttention(n_embedding, n_layer, n_head=12, mm=False)
        self.transform = nn.Linear(n_embedding, vocabulary_size, bias=False)
        self.transform.weight = self.embedding.weight
        self.prediction = nn.Sequential(nn.Linear(n_embedding, n_embedding // 2),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Linear(n_embedding // 2, 1, bias=False))

    def forward(self, inputs):
        [seq, index, target, all_pos, answer_pos] = inputs
        embedding = self.embedding(seq)
        encoder_representations, _ = self.encoder(embedding)
        encoder_representations = self.attention(encoder_representations)
        hidden = torch.masked_select(encoder_representations, all_pos.unsqueeze(2)).view(-1,
                                                                                         encoder_representations.size(
                                                                                             -1))
        end_score = self.prediction(hidden).squeeze(1)

        end_loss = F.binary_cross_entropy_with_logits(end_score, answer_pos)

        hidden = encoder_representations.gather(1, index.unsqueeze(2).expand(index.size(0), index.size(1),
                                                                             self.n_embedding))
        mask_loss = F.cross_entropy(F.log_softmax(self.transform(hidden.contiguous().view(-1, self.n_embedding)), 1),
                                    target.contiguous().view(-1))

        return end_loss, mask_loss

    def inference(self, seq):
        word_embedding = self.embedding(seq)
        encoder_representations, _ = self.encoder(word_embedding)
        encoder_representations = self.attention(encoder_representations)
        return encoder_representations
