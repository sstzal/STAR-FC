#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.utils import GraphConv, MeanAggregator
import datetime
import torch.optim as optim
import random

class GCN(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        #self.conv2 = GraphConv(nhid, nhid, MeanAggregator, dropout)
        #self.conv3 = GraphConv(nhid, nhid, MeanAggregator, dropout)

    def forward(self, data):
        x, adj = data[0], data[1]

        x = self.conv1(x, adj)
        #x = F.relu(self.conv2(x, adj)+x)
        #x = F.relu(self.conv3(x, adj) + x)

        return x

class HEAD(nn.Module):
    def __init__(self, nhid, dropout=0):
        super(HEAD, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(nn.Linear(nhid*2, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, 2))
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, feature, data, select_id=None):
        adj, label = data[1], data[2]
        feature = feature.view(-1,self.nhid)
        inst = adj._indices().size()[1]
        select_id = select_id

        if select_id is None:
            print('Dont have to select id.')
            row = adj._indices()[0,:]
            col = adj._indices()[1,:]
        else:
            row = adj._indices()[0, select_id].tolist()
            col = adj._indices()[1, select_id].tolist()
        patch_label = (label[row] == label[col]).long()
        pred = self.classifier(torch.cat((feature[row],feature[col]),1))

        loss = self.loss(pred, patch_label)
        return loss 

class HEAD_test(nn.Module):
    def __init__(self, nhid):
        super(HEAD_test, self).__init__()

        self.nhid = nhid
        self.classifier = nn.Sequential(nn.Linear(nhid*2, nhid), nn.PReLU(nhid),
                                        nn.Linear(nhid, 2),nn.Softmax(dim=1))

    def forward(self, feature1, feature2, no_list=False):
        if len(feature1.size())==1:
            pred = self.classifier(torch.cat((feature1,feature2)).unsqueeze(0))
            if pred[0][0]>pred[0][1]:
                is_same = False
            else:
                is_same = True
            return is_same
        else:
            pred = self.classifier(torch.cat((feature1,feature2),1))
            #print(pred[0:10,:])
            if no_list:
                return pred[:,1]
            score = list(pred[:,1].cpu().detach().numpy())
            #is_same = (pred[:,0]<pred[:,1]).long()

        return score

    
class EncoderBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        #pdb.set_trace()
        norm_x = self.norm1(x)
        x_1 = self.attn(norm_x)

        x = x + self.drop_path(x_1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x    
    
class select_encoder(nn.Module):
    def __init__(self, dim, multi=2, depth=1, heads_num=4, emb_dropout = 0.):
        super(select_encoder, self).__init__()

        self.transformer_encoder = EncoderBlock(dim, heads_num)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        b, n, c = x.shape

        x = self.transformer_encoder(x)
        x = self.mlp_head(x).squeeze(2)#b n
        x = F.gumbel_softmax(x, hard=True)
        x = x.unsqueeze(2)#b n 1

        return x#b n 1
    
class Distance_learner(nn.Module):
    def __init__(self):
        super(Distance_learner, self).__init__()
        init_p = torch.zeros(6)+0.0001
        init_p[3] = 0.1
        self.select_p = nn.Parameter(init_p)

    def forward(self, seed):
        a, idx = torch.sort(self.select_p, descending=True)
        torch.manual_seed(seed%50)
        p = F.gumbel_softmax(self.select_p, hard=True)
        best_index = torch.argmax(p)
        if idx[1]!=best_index:
            second_index = idx[1]
        else:
            second_index = idx[2]
        p_select=[]
        p_select.append(p[best_index])
        p_select.append(p[second_index])

        return best_index, second_index, p_select    

    
def gcn(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
