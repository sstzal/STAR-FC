from __future__ import division

from collections import OrderedDict

import torch
from src.datasets import build_dataset
from utils import sparse_mx_to_torch_sparse_tensor, build_knns,fast_knns2spmat,build_symmetric_adj,row_normalize,\
    Timer,read_probs,l2norm,read_meta,intdict2ndarray, build_knns_dynamic
import logging
import numpy as np
import torch.optim as optim
from src.models.gcn_v import GCN_V,HEAD, HEAD_cat, HEAD_test, select_encoder, Distance_learner
import datetime
import os
import linecache
import random
import faiss
from PIL import Image
import torchvision.transforms as transforms
from operator import itemgetter
from data import ImageFolder
from collections import OrderedDict
import time

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)

def tensor_l2norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def perform_val(model, HEAD1, HEAD_test1, cfg, feature_dim, pair_a, pair_b):

    test_lb2idxs, test_idx2lb = read_meta(cfg.test_data['label_path'])
    test_inst_num = len(test_idx2lb)

    model.eval()
    HEAD1.eval()
    HEAD_test1.eval()

    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.model['type'], cfg.test_data)

    features = torch.FloatTensor(dataset.features)
    adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
    labels = torch.LongTensor(dataset.gt_labels)

    if cfg.cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        HEAD_test1 = HEAD_test1.cuda()

    test_data = [features, adj, labels]

    HEAD_test1.load_state_dict(HEAD1.state_dict(), False)

    with torch.no_grad():
        output_feature = model(test_data)
        sum_acc = 0
        patch_num = 10
        patch_size = int(test_inst_num / patch_num)
        for i in range(patch_num):
            score = HEAD_test1(output_feature[pair_a[i * patch_size:(i + 1) * patch_size]],
                               output_feature[pair_b[i * patch_size:(i + 1) * patch_size]], no_list=True)
            #print(score)
            pre_labels = (score > 0.5).long()
            #print(pre_labels)
            gt_labels = (labels[pair_a[i * patch_size:(i + 1) * patch_size]] == labels[pair_b[i * patch_size:(i + 1) * patch_size]]).long()

            acc = (pre_labels == gt_labels).long().sum()
            sum_acc += acc
        avg_acc = float(sum_acc) / test_inst_num
        return avg_acc

def train_gcn_hierarchical(model, cfg, logger):
    # prepare dataset
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.train_data, k, v)
    dataset = build_dataset(cfg.model['type'], cfg.train_data)
    pre_features = torch.FloatTensor(dataset.features)
    print('Have loaded the training data.')

    inst_num = dataset.inst_num
    feature_dim = dataset.feature_dim
    lb2idxs = dataset.lb2idxs
    center_fea = dataset.center_fea.astype('float32')
    cls_num, dim = center_fea.shape
    labels = torch.LongTensor(dataset.gt_labels)

    MODEL_ROOT = './src/train_model_'+str(cfg.k_num2)
    if not os.path.exists(MODEL_ROOT):
        os.makedirs(MODEL_ROOT)
    print('the model save path is', MODEL_ROOT)


    HEAD1 = HEAD(nhid=512)
    HEAD2 = HEAD(nhid=512)
    HEAD3 = HEAD(nhid=512)
    HEAD4 = HEAD(nhid=512)
    select_attention = select_encoder(512)#select encoder
    distance_learner = Distance_learner()

    OPTIMIZER = optim.SGD([{'params': model.parameters(),'weight_decay':1e-5,'lr':0.01},
                           {'params': HEAD1.parameters(),'weight_decay':1e-5,'lr':0.01},
                           {'params': HEAD2.parameters(),'weight_decay':1e-5,'lr':0.01},
                           {'params': HEAD3.parameters(),'weight_decay':1e-5,'lr':0.01},
                           {'params': HEAD4.parameters(), 'weight_decay': 1e-5, 'lr': 0.01},
                           {'params': select_attention.parameters(),'weight_decay':1e-5,'lr':0.01},], momentum=0.9)
    OPTIMIZER2 = optim.SGD([{'params': distance_learner.parameters(),'weight_decay':1e-5,'lr':0.001},], momentum=0.9)
    print('the learning rate is 0.01')


    cfg.cuda = True
    model = model.cuda()
    HEAD1 = HEAD1.cuda()
    HEAD2 = HEAD2.cuda()
    HEAD3 = HEAD3.cuda()
    HEAD4 = HEAD4.cuda()
    select_attention = select_attention.cuda()
    distance_learner = distance_learner.cuda()


    for epoch in range(cfg.total_epochs):
        if epoch == cfg.STAGES[0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == cfg.STAGES[1]:
            schedule_lr(OPTIMIZER)

        model.train()
        HEAD1.train()
        HEAD2.train()
        HEAD3.train()
        HEAD4.train()
        select_attention.train()
        distance_learner.train()

        index = faiss.IndexFlatIP(dim)
        index.add(center_fea)
        sims, cluster_id = index.search(center_fea, k=1150)  # search for the k-10 neighbor
        print('the k is 1500')
        print(cluster_id.shape)
        print('Have selected the cluster ids.')


        per_class = list(range(cls_num))
        batch_num = 0
        for ind, batch in enumerate(per_class):
            #0.select ids
            sample_cluster_id = list(cluster_id[batch])
            sample_id = []#the idx of the samples in this batch
            sele = random.sample(sample_cluster_id, 1000)
            for i in sele:
                sample_id.extend(random.sample(lb2idxs[i],int(len(lb2idxs[i])*0.9)))

            sample_num =len(sample_id)

            if (sample_num>70000)|(sample_num<100):#55000
                print('[too much samples] continue.')
                continue

            #1.create selected labels and images
            batch_labels = labels[sample_id]
            feature = pre_features[sample_id]
            print(sample_num)

            #2.create knn for this batch
            with Timer('build knn:'):
                knn_prefix = os.path.join("./data/rebuild_knn")

                best_index, second_index, p_select = distance_learner(batch_num+20)
                max_value, max_index = torch.sort(torch.tensor([best_index,second_index]), descending = True)
                max_k1 = int((max_value[0] + 1) * 5)
                max_k2 = int((max_value[1] + 1) * 5)
                p1 = p_select[max_index[0]]
                p2 = p_select[max_index[1]]
                print('max_k1, max_k2, middle', max_k1, max_k2, cfg.k_num2)

                batch_adj_small1, batch_adj_small2, batch_adj_middle, batch_adj_large1, batch_adj_large2 =\
                    build_knns_dynamic(knn_prefix,
                               l2norm(feature.numpy()),
                               "faiss",
                               cfg.k_num2 + max_k1, max_k1, max_k2,
                               is_rebuild=True, save_knn=False)

                batch_adj_small1 = fast_knns2spmat(batch_adj_small1, cfg.k_num2 - max_k1, 0, use_sim=True)
                batch_adj_small1 = build_symmetric_adj(batch_adj_small1, self_loop=True)
                batch_adj_small1 = row_normalize(batch_adj_small1)
                batch_adj_small1 = sparse_mx_to_torch_sparse_tensor(batch_adj_small1, return_idx=False)

                batch_adj_small2 = fast_knns2spmat(batch_adj_small2, cfg.k_num2 - max_k2, 0, use_sim=True)
                batch_adj_small2 = build_symmetric_adj(batch_adj_small2, self_loop=True)
                batch_adj_small2 = row_normalize(batch_adj_small2)
                batch_adj_small2 = sparse_mx_to_torch_sparse_tensor(batch_adj_small2, return_idx=False)

                batch_adj_middle = fast_knns2spmat(batch_adj_middle, cfg.k_num2, 0, use_sim=True)
                batch_adj_middle = build_symmetric_adj(batch_adj_middle, self_loop=True)
                batch_adj_middle = row_normalize(batch_adj_middle)
                batch_adj_middle = sparse_mx_to_torch_sparse_tensor(batch_adj_middle, return_idx=False)

                batch_adj_large1 = fast_knns2spmat(batch_adj_large1, cfg.k_num2 + max_k1, 0, use_sim=True)
                batch_adj_large1 = build_symmetric_adj(batch_adj_large1, self_loop=True)
                batch_adj_large1 = row_normalize(batch_adj_large1)
                batch_adj_large1 = sparse_mx_to_torch_sparse_tensor(batch_adj_large1, return_idx=False)

                batch_adj_large2 = fast_knns2spmat(batch_adj_large2, cfg.k_num2 + max_k2, 0, use_sim=True)
                batch_adj_large2 = build_symmetric_adj(batch_adj_large2, self_loop=True)
                batch_adj_large2 = row_normalize(batch_adj_large2)
                batch_adj_large2 = sparse_mx_to_torch_sparse_tensor(batch_adj_large2, return_idx=False)

                batch_adj_small1 = batch_adj_small1.cuda()
                batch_adj_small2 = batch_adj_small2.cuda()
                batch_adj_middle = batch_adj_middle.cuda()
                batch_adj_large1 = batch_adj_large1.cuda()
                batch_adj_large2 = batch_adj_large2.cuda()


            #3.put selected feature and labels to cuda
            batch_labels = batch_labels.cuda()
            feature = feature.cuda()

            #4.train the model
            train_id_inst1 = batch_adj_small1._indices().size()[1]
            train_id_inst2 = batch_adj_middle._indices().size()[1]
            train_id_inst3 = batch_adj_large1._indices().size()[1]

            rad_id1 = np.random.choice(range(0, train_id_inst1),train_id_inst1,replace=False).tolist()+\
                      np.random.choice(range(0, train_id_inst1),train_id_inst1,replace=False).tolist()
            rad_id2 = np.random.choice(range(0, train_id_inst2),train_id_inst2,replace=False).tolist()+\
                      np.random.choice(range(0, train_id_inst2),train_id_inst2,replace=False).tolist()
            rad_id3 = np.random.choice(range(0, train_id_inst3),train_id_inst3,replace=False).tolist()

            patch_size = 120000
            patch_num = int(train_id_inst3/patch_size)
            print(patch_num)
            OPTIMIZER2.zero_grad()
            for i in range(patch_num):
                if i>0:
                    best_index, second_index, p_select = distance_learner(batch_num+20)
                    max_value, max_index = torch.sort(torch.tensor([best_index,second_index]), descending = True)
                    p1 = p_select[max_index[0]]
                    p2 = p_select[max_index[1]]

                id1 = rad_id1[i * patch_size:(i + 1) * patch_size]
                id2 = rad_id2[i * patch_size:(i + 1) * patch_size]
                id3 = rad_id3[i * patch_size:(i + 1) * patch_size]

                x_small1 = model([feature, batch_adj_small1, batch_labels])
                x_middle = model([feature, batch_adj_middle, batch_labels])
                x_large1 = model([feature, batch_adj_large1, batch_labels])
                x_small2 = model([feature, batch_adj_small2, batch_labels])
                x_large2 = model([feature, batch_adj_large2, batch_labels])
                x_small = p1 * x_small1 + p2 * x_small2
                x_large = p1 * x_large1 + p2 * x_large2
                x = select_attention(torch.cat((x_small.unsqueeze(1), x_middle.unsqueeze(1), x_large.unsqueeze(1)), 1))#B 3 C-> B 3 1
                select_ratio = x.sum(0)/ (x.sum())
                entropy = -(select_ratio[0]*torch.log2(select_ratio[0])+select_ratio[1]*torch.log2(select_ratio[1])+select_ratio[2]*torch.log2(select_ratio[2]))
                entropy = 1.5851 - entropy

                x = x.transpose(1,2) @ torch.cat((x_small.unsqueeze(1), x_middle.unsqueeze(1), x_large.unsqueeze(1)), 1)#b 1 c
                x = x.squeeze(1)#b c


                loss1 = HEAD1(x_small, [feature, batch_adj_small1, batch_labels], id1)
                loss2 = HEAD2(x_middle, [feature, batch_adj_middle, batch_labels], id2)
                loss3 = HEAD3(x_large, [feature, batch_adj_large1, batch_labels], id3)
                loss4 = HEAD4(x, [feature, batch_adj_large1, batch_labels], id3)
                loss = loss1 + loss2 + loss3 + loss4 + lamda * entropy

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()
                #print(datetime.datetime.now())
                print('epoch:{}/{}, batch:{}/{}, batch2:{}/{},loss:{}'.format(epoch, cfg.total_epochs, ind, cls_num, i, patch_num, loss))

            OPTIMIZER2.step()
            OPTIMIZER2.zero_grad()

            if batch_num%100==0:
                if not os.path.exists(MODEL_ROOT):
                    os.makedirs(MODEL_ROOT)
                print('save model in epoch:{} batch:{} to {}'.format(epoch, batch_num, MODEL_ROOT))
                if finetune:
                    torch.save(model.state_dict(), os.path.join(MODEL_ROOT, "Backbone1_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(HEAD1.state_dict(), os.path.join(MODEL_ROOT, "Head1_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(HEAD2.state_dict(), os.path.join(MODEL_ROOT, "Head2_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(HEAD3.state_dict(), os.path.join(MODEL_ROOT, "Head3_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(HEAD4.state_dict(), os.path.join(MODEL_ROOT, "Head4_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(select_attention.state_dict(), os.path.join(MODEL_ROOT, "Attention_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(distance_learner.state_dict(), os.path.join(MODEL_ROOT, "Distance_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                else:
                    torch.save(HEAD4.state_dict(), os.path.join(MODEL_ROOT, "Head4_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
                    torch.save(select_attention.state_dict(), os.path.join(MODEL_ROOT, "Attention_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
            batch_num = batch_num + 1

        #5.test
        #avg_acc = perform_val(model, HEAD1, HEAD_test1, cfg, feature_dim, pair_a, pair_b)
        #print('the avg testing acc in epoch:{} batch:{} is :'.format(epoch,batch), avg_acc)


        # 6.save model
        if not os.path.exists(MODEL_ROOT):
            os.makedirs(MODEL_ROOT)
        print('save model in epoch:{} batch:{} to {}'.format(epoch, ind, MODEL_ROOT))
        torch.save(model.state_dict(),
                   os.path.join(MODEL_ROOT, "Backbone1_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(HEAD1.state_dict(),
                   os.path.join(MODEL_ROOT, "Head1_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(HEAD2.state_dict(),
                   os.path.join(MODEL_ROOT, "Head2_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(HEAD3.state_dict(),
                   os.path.join(MODEL_ROOT, "Head3_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(HEAD4.state_dict(),
                   os.path.join(MODEL_ROOT, "Head4_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(select_attention.state_dict(),
                   os.path.join(MODEL_ROOT, "Attention_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
        torch.save(distance_learner.state_dict(),
                   os.path.join(MODEL_ROOT, "Distance_Epoch_{}_batch_{}.pth".format(epoch + 1, batch_num)))
