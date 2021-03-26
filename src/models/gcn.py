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


def gcn(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
