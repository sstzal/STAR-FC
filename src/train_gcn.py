from __future__ import division

from collections import OrderedDict

import torch
from src.datasets import build_dataset
from utils import sparse_mx_to_torch_sparse_tensor, build_knns,fast_knns2spmat,build_symmetric_adj,row_normalize,\
    Timer,read_probs,l2norm,read_meta,intdict2ndarray
import numpy as np
import torch.optim as optim
from src.models.gcn import HEAD, HEAD_test
import datetime
import os
import linecache
import random
import faiss
from PIL import Image
import torchvision.transforms as transforms
from operator import itemgetter
from collections import OrderedDict

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

def train_gcn(model, cfg, logger):
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
    HEAD1 = HEAD(nhid=512)
    HEAD_test1 = HEAD_test(nhid=512)

    #load parameters from the pretrained model
    #model.load_state_dict(torch.load('./'))
    #HEAD1.load_state_dict(torch.load('./'), False)

    OPTIMIZER = optim.SGD([{'params': model.parameters(),'weight_decay':1e-5},
                           {'params': HEAD1.parameters(),'weight_decay':1e-5}], lr=0.01, momentum=0.9)
    print('the learning rate is 0.01')

    #model.load_state_dict(torch.load(''))
    #HEAD1.load_state_dict(torch.load(''))
    print("have load the pretrained model.")
    cfg.cuda = True
    model = model.cuda()
    HEAD1 = HEAD1.cuda()

    MODEL_ROOT = './src/train_model'
    print('the model save path is', MODEL_ROOT)

    #prepare the test data
    target = "part1_test"
    knn_path = "./data/knns/" + target + "/faiss_k_80.npz"
    knns = np.load(knn_path, allow_pickle=True)['data']
    inst_num = knns.shape[0]
    k_num = knns.shape[2]
    nbrs = knns[:, 0, :]
    pair_a = []
    pair_b = []
    for i in range(inst_num):
        pair_a.extend([i] * k_num)
        pair_b.extend(nbrs[i])


    for epoch in range(cfg.total_epochs):
        if epoch == cfg.STAGES[0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
            schedule_lr(OPTIMIZER)
        if epoch == cfg.STAGES[1]:
            schedule_lr(OPTIMIZER)
        if epoch == cfg.STAGES[2]:
            schedule_lr(OPTIMIZER)

        model.train()
        HEAD1.train()

        index = faiss.IndexFlatIP(dim)
        index.add(center_fea)
        sims, cluster_id = index.search(center_fea, k=(cfg.cluster_num+200))  # search for the k-10 neighbor
        #sims, cluster_id = index.search(center_fea, k=cfg.cluster_num)  # search for the k-10 neighbor
        print('Have selected the cluster ids.')

        for batch in range(cls_num):
        #for batch in range(20):
            #0.select ids
            sample_cluster_id = random.sample(list(cluster_id[batch]), cfg.cluster_num)
            #sample_cluster_id = list(cluster_id[batch])
            sample_id = []#the idx of the samples in this batch
            for i in range(len(sample_cluster_id)):
                sample_id.extend(random.sample(lb2idxs[sample_cluster_id[i]],int(len(lb2idxs[sample_cluster_id[i]])*0.9)))
                #sample_id.extend(lb2idxs[sample_cluster_id[i]])
            #sample_id.sort()
            sample_num =len(sample_id)
            #id = list(np.arange(0,sample_num,1))
            #sample2sort = dict(zip(sample_id, id))
            if (sample_num>100000)|(sample_num<100):
                print('[too much samples] continue.')
                continue

            #1.create selected labels and images
            batch_labels = labels[sample_id]
            feature = pre_features[sample_id]
            print(sample_num)

            #2.create knn for this batch
            with Timer('build knn:'):
                knn_prefix = os.path.join("./data/rebuild_knn")
                if not os.path.exists(knn_prefix):
                    os.makedirs(knn_prefix)
                if os.path.exists(os.path.join(knn_prefix, 'faiss_k_%d.npz' % cfg.knn)):
                    os.remove(os.path.join(knn_prefix, 'faiss_k_%d.npz' % cfg.knn))
                if os.path.exists(os.path.join(knn_prefix, 'faiss_k_%d.index' % cfg.knn)):
                    os.remove(os.path.join(knn_prefix, 'faiss_k_%d.index' % cfg.knn))

                knns = build_knns(knn_prefix,
                                  #l2norm(feature.clone().detach().cpu().numpy()),
                                  l2norm(feature.numpy()),
                                  cfg.knn_method,  # "faiss"
                                  cfg.knn,  # 80
                                  is_rebuild=True)
                batch_adj = fast_knns2spmat(knns, cfg.knn, 0, use_sim=True)
                batch_adj = build_symmetric_adj(batch_adj, self_loop=True)
                batch_adj = row_normalize(batch_adj)
                batch_adj = sparse_mx_to_torch_sparse_tensor(batch_adj, return_idx=False)

            #3.put selected feature and labels to cuda
            batch_labels = batch_labels.cuda()
            feature = feature.cuda()
            batch_adj = batch_adj.cuda()
            train_data = [feature, batch_adj, batch_labels]
            #x = model(train_data)

            #4.train the model
            #add
            train_id_inst = batch_adj._indices().size()[1]
            #print('train_id_inst:', train_id_inst)
            #print('sample_num:', sample_num)
            #train_id_inst = sample_num
            rad_id = random.sample(range(0, train_id_inst), train_id_inst)+random.sample(range(0, train_id_inst), train_id_inst)
            patch_num = 40
            for i in range(patch_num*2):
                id = rad_id[i * int(train_id_inst / patch_num):(i + 1) * int(train_id_inst / patch_num)]
                x = model(train_data)
                loss = HEAD1(x, train_data, id)

                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                print(datetime.datetime.now())
                print('epoch:{}/{}, batch:{}/{}, batch2:{}/{},loss:{}'.format(epoch, cfg.total_epochs, batch, cls_num, i, patch_num*2, loss))

            if (batch+1)%100==0:
                if not os.path.exists(MODEL_ROOT):
                    os.makedirs(MODEL_ROOT)
                print('save model in epoch:{} batch:{} to {}'.format(epoch, batch, MODEL_ROOT))
                torch.save(model.state_dict(), os.path.join(MODEL_ROOT, "Backbone_Epoch_{}_batch_{}.pth".format(epoch + 1, batch)))
                torch.save(HEAD1.state_dict(), os.path.join(MODEL_ROOT, "Head_Epoch_{}_batch_{}.pth".format(epoch + 1, batch)))
            
            if (batch + 1) % 300 == 0:
                avg_acc = perform_val(model, HEAD1, HEAD_test1, cfg, feature_dim, pair_a, pair_b)
                print('the avg testing acc in epoch:{} batch:{} is :'.format(epoch,batch), avg_acc)
                model.train()
                HEAD1.train()


        #5.test
        avg_acc = perform_val(model, HEAD1, HEAD_test1, cfg, feature_dim, pair_a, pair_b)
        print('the avg testing acc in epoch:{} batch:{} is :'.format(epoch,batch), avg_acc)


        # 6.save model
        if not os.path.exists(MODEL_ROOT):
            os.makedirs(MODEL_ROOT)
        print('save model in epoch:{} batch:{} to {}'.format(epoch, batch, MODEL_ROOT))
        torch.save(model.state_dict(), os.path.join(MODEL_ROOT, "Backbone_Epoch_{}_batch_{}.pth".format(epoch + 1, batch)))
        torch.save(HEAD1.state_dict(), os.path.join(MODEL_ROOT, "Head_Epoch_{}_batch_{}.pth".format(epoch + 1, batch)))
