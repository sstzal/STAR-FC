import numpy as np
import scipy.sparse as sp
from utils import Timer
import datetime
import torch
import random
import pdb
from scipy.sparse import csr_matrix
from mmcv import Config
import os
from src.models import build_model
from src.datasets import build_dataset
import linecache
from src.models.gcn import HEAD, HEAD_test, select_encoder
from tqdm import tqdm
import time
from utils import sparse_mx_to_torch_sparse_tensor, build_knns, fast_knns2spmat, build_symmetric_adj, row_normalize,mkdir_if_no_exists, indices_values_to_sparse_tensor, l2norm, read_probs
from evaluation.evaluate import evaluate
from scipy.sparse import csr_matrix
from graph import graph_propagation, connected_components_constraint
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics

def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id


def step1(target = 'part1_test', model_path = 'train_model_100', backbone_index = ['4300'], k_num=120):
    cfg = Config.fromfile("./src/configs/cfg_gcn_ms1m_hierarchical.py")
    cfg.eval_interim = False

    feature_path = "./data/features"

    for model_i in [0]:
        model_i = int(model_i)
        model_path2 = model_path
        model_path = "./src/" + model_path
        print('model_path', model_path)
        backbone_name = "Backbone1_Epoch_2_batch_" + backbone_index[model_i] + ".pth"
        HEAD_name = "Head4_Epoch_2_batch_" + backbone_index[model_i] + ".pth"
        attention_name = "Attention_Epoch_2_batch_" + backbone_index[model_i] + ".pth"
        use_cuda = False
        knn_path = "./data/knns/part1_test/faiss_k_"+str(k_num)+".npz"
        use_gcn = True

        if use_gcn:
            knns = np.load(knn_path, allow_pickle=True)['data']
            nbrs = knns[:, 0, :]
            dists = knns[:, 1, :]
            edges = []
            score = []
            inst_num = knns.shape[0]
            print("inst_num:", inst_num)

            model = build_model('gcn_v', **cfg.model['kwargs'])
            model.load_state_dict(torch.load(os.path.join(model_path, backbone_name)))
            HEAD_test1 = HEAD_test(nhid=512)
            HEAD_test1.load_state_dict(torch.load(os.path.join(model_path, HEAD_name)), False)
            slot_attention = select_encoder(512)
            slot_attention.load_state_dict(torch.load(os.path.join(model_path, attention_name)))


            with Timer('build dataset'):
                for k, v in cfg.model['kwargs'].items():
                    setattr(cfg.test_data, k, v)
                dataset = build_dataset(cfg.model['type'], cfg.test_data)

            features = torch.FloatTensor(dataset.features)
            adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
            if not dataset.ignore_label:
                labels = torch.FloatTensor(dataset.gt_labels)

            #add knn with different k
            knn_prefix = os.path.join("./data/knns/part1_test")


            knns_80 = build_knns(knn_prefix,
                                 l2norm(features.numpy()),
                                 "faiss",
                                 80,
                                 is_rebuild=False)
            adj_80 = fast_knns2spmat(knns_80, 80, 0, use_sim=True)
            adj_80 = build_symmetric_adj(adj_80, self_loop=True)
            adj_80 = row_normalize(adj_80)
            adj_80 = sparse_mx_to_torch_sparse_tensor(adj_80, return_idx=False)


            knns_100 = build_knns(knn_prefix,
                                 l2norm(features.numpy()),
                                 "faiss",
                                 100,
                                 is_rebuild=False)
            adj_100 = fast_knns2spmat(knns_100, 100, 0, use_sim=True)
            adj_100 = build_symmetric_adj(adj_100, self_loop=True)
            adj_100 = row_normalize(adj_100)
            adj_100 = sparse_mx_to_torch_sparse_tensor(adj_100, return_idx=False)

            knns_120 = build_knns(knn_prefix,
                                 l2norm(features.numpy()),
                                 "faiss",
                                 120,
                                 is_rebuild=False)
            adj_120 = fast_knns2spmat(knns_120, 120, 0, use_sim=True)
            adj_120 = build_symmetric_adj(adj_120, self_loop=True)
            adj_120 = row_normalize(adj_120)
            adj_120 = sparse_mx_to_torch_sparse_tensor(adj_120, return_idx=False)


            if k_num==80:
                pair_a_new = adj._indices()[0].int().tolist()
                pair_b_new = adj._indices()[1].int().tolist()
            elif k_num==100:
                pair_a_new = adj_100._indices()[0].int().tolist()
                pair_b_new = adj_100._indices()[1].int().tolist()
            elif k_num==120:
                pair_a_new = adj_120._indices()[0].int().tolist()
                pair_b_new = adj_120._indices()[1].int().tolist()
            print(len(pair_a_new))
            inst_num = len(pair_a_new)


            if use_cuda:
                model.cuda()
                HEAD_test1.cuda()
                slot_attention.cuda()
                features = features.cuda()
                #adj = adj.cuda()
                adj_100 = adj_100.cuda()
                adj_80 = adj_80.cuda()
                adj_120 = adj_120.cuda()
                labels = labels.cuda()

            model.eval()
            HEAD_test1.eval()
            slot_attention.eval()

            score = []
            for threshold1 in [0.85]:
                with torch.no_grad():
                    with Timer('gcn1'):
                        output_feature_80 = model([features, adj_80, labels])
                    with Timer('gcn2'):
                        output_feature_100 = model([features, adj_100, labels])
                    with Timer('gcn3'):
                        output_feature_120 = model([features, adj_120, labels])
                    with Timer('Selector:'):
                        x = slot_attention(
                            torch.cat((output_feature_80.unsqueeze(1), output_feature_100.unsqueeze(1),
                                       output_feature_120.unsqueeze(1)), 1))  # B 3 C-> B 3 1
                        x = x.transpose(1, 2) @ \
                            torch.cat((output_feature_80.unsqueeze(1), output_feature_100.unsqueeze(1),
                                       output_feature_120.unsqueeze(1)), 1)  # b 1 c
                    x = x.squeeze(1)  # b c
                    patch_num = 65
                    patch_size = int(inst_num / patch_num)
                    for i in range(patch_num):
                        id1 = pair_a_new[i * patch_size:(i + 1) * patch_size]
                        id2 = pair_b_new[i * patch_size:(i + 1) * patch_size]
                        score_ = HEAD_test1(x[id1], x[id2])
                        score_ = np.array(score_)
                        idx = np.where(score_ > threshold1)[0].tolist()

                        id1 = np.array(id1)
                        id2 = np.array(id2)
                        id1 = np.array([id1[idx].tolist()])
                        id2 = np.array([id2[idx].tolist()])
                        edges.extend(np.concatenate([id1, id2], 0).transpose().tolist())
                    id1 = pair_a_new[(i+1) * patch_size:]
                    id2 = pair_b_new[(i+1) * patch_size:]
                    score_ = HEAD_test1(x[id1], x[id2])
                    score_ = np.array(score_)
                    idx = np.where(score_ > threshold1)[0].tolist()

                    id1 = np.array(id1)
                    id2 = np.array(id2)
                    id1 = np.array([id1[idx].tolist()])
                    id2 = np.array([id2[idx].tolist()])
                    edges.extend(np.concatenate([id1, id2], 0).transpose().tolist())

                edges = np.array(edges)

    return edges


def compute_ni_faster(edges):
    inst_num = 584013#584013,1740301,2890517,4046365,5206761,
    edges = np.sort(edges)
    edges = np.unique(edges, axis=0)

    row_ = edges[:, 0].tolist()
    col_ = edges[:, 1].tolist()
    row = row_ + col_
    col = col_ + row_
    value = [1] * (len(edges) * 2)

    adj = csr_matrix((value, (row, col)), shape=(inst_num, inst_num))

    row = np.arange(0,inst_num,1)
    col = row
    value = adj.diagonal(0)
    adj_diag = csr_matrix((value, (row, col)), shape=(inst_num, inst_num))
    adj = adj - adj_diag

    link_num = np.array(adj.sum(axis=1))
    #pdb.set_trace()
    neibor1 = link_num[edges[:, 0]]
    neibor2 = link_num[edges[:, 1]]

    adj2 = adj.dot(adj)
    share_num = np.array(adj2[edges[:, 0].tolist(), edges[:, 1].tolist()].tolist())
    share_num = share_num.reshape((-1, 1))
    ni = np.maximum(share_num/neibor1, share_num/neibor2)
    ni[np.isnan(ni)] = 0

    return edges, ni, neibor1, neibor2 #the ni


def label2idx(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb


def compute_ni_faster_dynamic(edges, ni, neibor1, neibor2):
    inst_num = 584013#584013,1740301,2890517,4046365,5206761,
    edges = np.sort(edges)
    edges = np.unique(edges, axis=0)

    gt_labels = np.load('./data/gt_label_926.npy')

    threshold2_list = [0.78]
    edges_new = []
    ni_modulate = ni

    for th2 in threshold2_list:
        edges_new = []
        for i in range(len(edges)):
            qujian = int(np.minimum(int(min(neibor1[i], neibor2[i]) / 10), 2)) + 1
            th3 = th2 / ((0.85)**(qujian*10)+1)
            if (ni_modulate[i] > th3) and (edges[i][0]!=edges[i][1]):
                edges_new.append(edges[i])

        edges_new = np.array(edges_new)
        row = edges_new[:, 0].tolist()
        col = edges_new[:, 1].tolist()
        value = [1] * len(edges_new)
        adj = csr_matrix((value, (row, col)), shape=(inst_num, inst_num))

        components, pre_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        evaluate(gt_labels, pre_labels, 'pairwise')
        evaluate(gt_labels, pre_labels, 'bcubed')
        evaluate(gt_labels, pre_labels, 'nmi')
    return edges, ni


def ev(edges, score_confidence3, model_path):
    threshold2_list=[0.7,0.72,0.74,0.76,0.78,0.8,0.82]
    for threshold2 in threshold2_list:
        edges_new = []
        for i in range(len(edges)):
            if score_confidence3[i] > threshold2:
                edges_new.append(edges[i])
            #if i % 10000000 == 0:
            #    print('part5 {}/{}'.format(i, len(edges)))

        #with Timer('find components2:'):
        edges_new = np.array(edges_new)
        row = edges_new[:, 0].tolist()
        col = edges_new[:, 1].tolist()
        value = [1] * len(edges_new)
        inst_num = 584013
        adj = csr_matrix((value, (row, col)), shape=(inst_num, inst_num))
        components, pre_labels = connected_components(csgraph=adj, directed=False, return_labels=True)
        #with Timer('find components3:'):
        #    pre_labels = edge_to_connected_graph(edges_new, 584013)
        #gt_labels = np.load('./data/gt_label_train.npy')
        gt_labels = np.load('./data/gt_label_926.npy')
        #print('the threshold1 is:{}'.format(threshold1))
        print('the threshold2 is:{}'.format(threshold2))
        evaluate(gt_labels, pre_labels, 'pairwise')
        #evaluate(gt_labels, pre_labels, 'bcubed')
        #evaluate(gt_labels, pre_labels, 'nmi')
        #pdb.set_trace()
    print(model_path)


def ev_gt(edges):
    gt_labels = np.load('./data/gt_label_926.npy')
    edges_new = []
    for i in range(len(edges)):
        if gt_labels[edges[i][0]] == gt_labels[edges[i][1]]:
            edges_new.append(edges[i])
        if i % 10000000 == 0:
            print('part5 {}/{}'.format(i, len(edges)))

    pre_labels = edge_to_connected_graph(edges_new, 584013)
    evaluate(gt_labels, pre_labels, 'pairwise')
    #evaluate(gt_labels, pre_labels, 'bcubed')
    #evaluate(gt_labels, pre_labels, 'nmi')
    #pdb.set_trace()
    print(model_path)


edges = step1()
edges, score, neibor1, neibor2 = compute_ni_faster(edges)
edges, score = compute_ni_faster_dynamic(edges, score, neibor1, neibor2)
