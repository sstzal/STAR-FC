import os.path as osp

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'
#train_name='all'
knn = 80
knn_method = 'faiss'
th_sim = 0.  
INPUT_SIZE = [112,112]
cluster_num = 1300
mid_feat_path = osp.join(prefix, 'mid_features','{}.bin'.format(train_name))
mid_feat_path_test = osp.join(prefix, 'mid_features_test','{}.bin'.format(train_name))
#all_img_path = osp.join(prefix, 'lists_add', '{}.list'.format(train_name)),
# if `knn_graph_path` is not passed, it will build knn_graph automatically
train_data = dict(feat_path=osp.join(prefix, 'features',
                                    '{}.bin'.format(train_name)),
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(train_name)),
                 knn_graph_path=osp.join(prefix, 'knns', train_name,
                                         '{}_k_{}.npz'.format(knn_method, knn)),
                 list_path=osp.join(prefix, 'lists', '{}.list'.format(train_name)),
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 conf_metric='s_nbr',
                 eval_interim = False)

test_data = dict(feat_path=osp.join(prefix, 'features','{}.bin'.format(test_name)),
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(test_name)),
                 knn_graph_path=osp.join(prefix, 'knns', test_name,
                                         '{}_k_{}.npz'.format(knn_method, knn)),
                 list_path=osp.join(prefix, 'lists', '{}.list'.format(test_name)),
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 conf_metric='s_nbr',
                 eval_interim = False)

# model
model = dict(type='gcn',
             kwargs=dict(feature_dim=256, nhid=512, nclass=1, dropout=0.))

# training args
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = {}

total_epochs = 50
STAGES = [int(total_epochs*0.5), int(total_epochs*0.8), int(total_epochs*0.9)]
lr_config = dict(
    policy='step',
    step = [int(r * total_epochs) for r in [0.5, 0.8, 0.9]]
)

batch_size_per_gpu = 1

# testing args
use_gcn_feat = True
eval_interim = False

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=5000)

log_level = 'INFO'
log_config = dict(interval=1, hooks=[
    dict(type='TextLoggerHook'),
])
