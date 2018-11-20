# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/20 上午10:37
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import torch.nn as nn
import torch
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self, n_input):
        super().__init__()

        self.conv_q = nn.Conv2d(in_channels=1, out_channels=n_input, kernel_size=(3, n_input), padding=(1, 0))
        self.conv_p = nn.Conv2d(in_channels=1, out_channels=n_input, kernel_size=(3, n_input), padding=(1, 0))

        self.q_U = nn.Linear(2 * n_input, n_input // 2)
        self.p_U = nn.Linear(2 * n_input, n_input // 2)
        self.v = nn.Linear(n_input // 2, n_input // 2)
        self.project = nn.Linear(n_input, n_input)

        nn.init.xavier_normal_(self.q_U.weight, gain=0.1)
        nn.init.xavier_normal_(self.p_U.weight, gain=0.1)
        nn.init.xavier_normal_(self.v.weight, gain=0.1)
        nn.init.xavier_normal_(self.project.weight, gain=0.1)

    def get_hidden(self, representations, cnn, linear):
        hidden = F.relu(cnn(representations.unsqueeze(1)).squeeze(3).transpose(2, 1), inplace=True)
        return F.leaky_relu(linear(torch.cat([hidden, representations], -1)), inplace=True)

    def forward(self, representations):
        s1 = self.get_hidden(representations, self.conv_q, self.q_U)
        s2 = self.get_hidden(representations, self.conv_p, self.p_U)
        score = F.softmax(self.v(s1).bmm(s2.transpose(2, 1)), 2)

        return representations + score.bmm(F.leaky_relu(self.project(representations), inplace=True))


class MultiHeadBlock(nn.Module):
    def __init__(self, n_input, n_head=6):
        super().__init__()
        self.combined_projection = nn.Linear(n_input, 2 * (n_input // n_head) * n_head + (n_input // 2) * n_head)
        self.output_projection = nn.Linear((n_input // 2) * n_head, n_input)

        nn.init.xavier_normal_(self.combined_projection.weight, gain=0.1)
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.1)

        self._scale = (n_input // n_head) ** 0.5
        self.att_dim = (n_input // n_head) * n_head
        self.num_heads = n_head

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

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        # Shape (num_heads * batch_size, timesteps, attention_dim / num_heads)
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

        return representations + F.leaky_relu(self.output_projection(outputs))


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6):
        super().__init__()
        self.n_head = n_head
        self.att = nn.ModuleList()
        for _ in range(n_layer):
            en = AttentionBlock(n_hidden)
            # en = AttentionBlockBeta(n_hidden)
            # en = MultiHeadBlock(n_hidden)
            ln = nn.LayerNorm(n_hidden)
            self.att.append(nn.Sequential(en, ln))

    def forward(self, representations):
        for one in self.att:
            representations = one(representations)
        return representations


class MultiHead(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 4, embedding_dim=n_embedding)
        self.n_embedding = n_embedding
        self.encoder = nn.GRU(input_size=n_embedding, hidden_size=n_hidden, batch_first=True,
                              bidirectional=True)
        self.projection = nn.Linear(2 * n_hidden, n_embedding)
        self.attention = SelfAttention(n_embedding, n_layer)
        self.att = nn.Linear(n_embedding, 1, bias=False)
        self.output = nn.Linear(n_embedding, 1, bias=False)
        self.mask_output = nn.AdaptiveLogSoftmaxWithLoss(in_features=n_embedding, n_classes=vocab_size + 4,
                                                         cutoffs=[520, 2048, 8503, 20832],
                                                         div_value=2)
        self.criterion = nn.BCELoss()

    def forward(self, seq):
        embedding = self.embedding(seq)
        encoder_representations, _ = self.encoder(embedding)
        encoder_representations = F.leaky_relu(self.projection(encoder_representations), inplace=True)
        encoder_representations = self.attention(encoder_representations)
        return encoder_representations
