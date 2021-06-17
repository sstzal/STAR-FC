import os
import numpy as np
import torch

from utils import (read_meta, read_probs, l2norm, build_knns,
                   knns2ordered_nbrs, fast_knns2spmat, row_normalize,
                   build_symmetric_adj, sparse_mx_to_indices_values,
                   intdict2ndarray, Timer)


class GCNDataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg.get('knn_graph_path', None)

        self.k = cfg['k']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)
        self.save_decomposed_adj = cfg.get('save_decomposed_adj', False)

        self.th_sim = cfg.get('th_sim', 0.)
        self.conf_metric = cfg.get('conf_metric')

        with Timer('read meta and feature'):
            if label_path is not None:
                self.lb2idxs, self.idx2lb = read_meta(label_path)
                self.inst_num = len(self.idx2lb)
                self.cls_num = len(self.lb2idxs)
                self.gt_labels = intdict2ndarray(self.idx2lb)
                self.ignore_label = False
            else:
                self.inst_num = -1
                self.ignore_label = True
            self.features = read_probs(feat_path, self.inst_num, self.feature_dim)

            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = 1  # take the entire graph as input

        with Timer('Compute center feature'):
            self.center_fea = np.zeros((self.cls_num, self.features.shape[1]))
            lbs = list(self.lb2idxs.keys())  # in case of uncontinuous ids
            for i in range(self.cls_num):
                _id = lbs[i]
                self.center_fea[i] = np.mean(self.features[self.lb2idxs[_id]], 0)
            self.center_fea = l2norm(self.center_fea)

        with Timer('read knn graph'):
            if os.path.isfile(knn_graph_path):
                print("load knns from the knn_path")
                self.knns = np.load(knn_graph_path)['data']
            else:
                if knn_graph_path is not None:
                    print('knn_graph_path does not exist: {}'.format(
                        knn_graph_path))
                knn_prefix = os.path.join(cfg.prefix, 'knns', cfg.name)
                self.knns = build_knns(knn_prefix, self.features, cfg.knn_method, cfg.knn)

            adj = fast_knns2spmat(self.knns, self.k, self.th_sim, use_sim=True)

            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            # print('adj before norm')
            # print(adj)
            adj = row_normalize(adj)
            if self.save_decomposed_adj:
                adj = sparse_mx_to_indices_values(adj)
                self.adj_indices, self.adj_values, self.adj_shape = adj
            else:
                self.adj = adj

            # convert knns to (dists, nbrs)
            self.dists, self.nbrs = knns2ordered_nbrs(self.knns)

        print('feature shape: {}, k: {}, norm_feat: {}'.format(
            self.features.shape, self.k, self.is_norm_feat))

    def __getitem__(self, index):

        assert index == 0
        return (self.features, self.adj_indices, self.adj_values,
                self.adj_shape, self.labels)

    def __len__(self):
        return self.size
