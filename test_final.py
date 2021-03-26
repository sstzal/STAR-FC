import torch
import linecache
import numpy as np
from evaluation.evaluate import evaluate
import os
from src.models.gcn import HEAD, HEAD_test
from src.models import build_model
from mmcv import Config
from src.datasets import build_dataset
from utils import sparse_mx_to_torch_sparse_tensor, build_knns, fast_knns2spmat, build_symmetric_adj, row_normalize,mkdir_if_no_exists,Timer
import torch.nn as nn
from utils.misc import l2norm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix



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

flag=1

if flag == 1:

	cfg = Config.fromfile("./src/configs/cfg_gcn_ms1m.py")
	cfg.eval_interim = False

	target = "part1_test"
	feature_path = "./data/features"

	#model_path_list=['train_model_sample7']
	#backbone_index=['4299']

	for model_i in [0]:
		model_i = int(model_i)
		model_path = "./pretrained_model"
		print('model_path',model_path)
		backbone_name = "Backbone.pth"
		HEAD_name = "Head.pth"
		use_cuda = True
		knn_path = "./data/knns/" + target + "/faiss_k_80.npz"
		use_gcn = True

		if use_gcn:
			knns = np.load(knn_path, allow_pickle=True)['data']
			nbrs = knns[:, 0, :]
			dists = knns[:, 1, :]
			edges = []
			score = []
			inst_num = knns.shape[0]
			print("inst_num:", inst_num)

			feature_path = os.path.join(feature_path, target)

			# print(**cfg.model['kwargs'])
			model = build_model('gcn', **cfg.model['kwargs'])
			model.load_state_dict(torch.load(os.path.join(model_path, backbone_name)))
			HEAD_test1 = HEAD_test(nhid=512)
			HEAD_test1.load_state_dict(torch.load(os.path.join(model_path, HEAD_name)), False)

			with Timer('build dataset'):
				for k, v in cfg.model['kwargs'].items():
					setattr(cfg.test_data, k, v)
				dataset = build_dataset(cfg.model['type'], cfg.test_data)

			features = torch.FloatTensor(dataset.features)
			adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
			if not dataset.ignore_label:
				labels = torch.FloatTensor(dataset.gt_labels)

			pair_a = []
			pair_b = []
			pair_a_new = []
			pair_b_new = []
			for i in range(inst_num):
				pair_a.extend([int(i)] * 80)
				pair_b.extend([int(j) for j in nbrs[i]])
			for i in range(len(pair_a)):
				if pair_a[i] != pair_b[i]:
					pair_a_new.extend([pair_a[i]])
					pair_b_new.extend([pair_b[i]])
			pair_a = pair_a_new
			pair_b = pair_b_new
			print(len(pair_a))
			inst_num = len(pair_a)
			if use_cuda:
				model.cuda()
				HEAD_test1.cuda()
				features = features.cuda()
				adj = adj.cuda()
				labels = labels.cuda()

			model.eval()
			HEAD_test1.eval()
			test_data = [[features, adj, labels]]

			for threshold1 in [0.7]:
				with Timer('Inference'):
					with Timer('First-0 step'):
						with torch.no_grad():
							output_feature = model(test_data[0])

							patch_num = 65
							patch_size = int(inst_num / patch_num)
							for i in range(patch_num):
								id1 = pair_a[i * patch_size:(i + 1) * patch_size]
								id2 = pair_b[i * patch_size:(i + 1) * patch_size]
								score_ = HEAD_test1(output_feature[id1],output_feature[id2])
								score_ = np.array(score_)
								idx = np.where(score_ > threshold1)[0].tolist()
								#score.extend(score_[idx].tolist())
								id1 = np.array(id1)
								id2 = np.array(id2)
								id1 = np.array([id1[idx].tolist()])
								id2 = np.array([id2[idx].tolist()])
								edges.extend(np.concatenate([id1, id2], 0).transpose().tolist())
								#print('patch id:',i)

						value=[1]*len(edges)
						edges=np.array(edges)

					with Timer('First step'):
						adj2 = csr_matrix((value, (edges[:,0].tolist(), edges[:,1].tolist())), shape=(584013, 584013))
						link_num = np.array(adj2.sum(axis=1))
						common_link = adj2.dot(adj2)

					for threshold2 in [0.72]:
						with Timer('Second step'):
							edges_new = []
							edges = np.array(edges)
							share_num = common_link[edges[:,0].tolist(), edges[:,1].tolist()].tolist()[0]
							edges = edges.tolist()

							for i in range(len(edges)):
								if ((link_num[edges[i][0]]) != 0) & ((link_num[edges[i][1]]) != 0):
									if max((share_num[i])/link_num[edges[i][0]],(share_num[i])/link_num[edges[i][1]])>threshold2:
										edges_new.append(edges[i])
								if i%10000000==0:
									print(i)

						with Timer('Last step'):
							pre_labels = edge_to_connected_graph(edges_new, 584013)
						gt_labels = np.load('./pretrained_model/gt_labels.npy')
						print('the threshold1 is:{}'.format(threshold1))
						print('the threshold2 is:{}'.format(threshold2))
						evaluate(gt_labels, pre_labels, 'pairwise')
						evaluate(gt_labels, pre_labels, 'bcubed')
						evaluate(gt_labels, pre_labels, 'nmi')

