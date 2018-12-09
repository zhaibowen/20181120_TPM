# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 9:32
# @Author  : LeeYun
# @Email   : leeyun.bw@gmail.com
# @File    : config.py
import numpy as np

processor = 11

# track params
acc = [0, 20, -20]  # three modes
T = 10  # time period
trans = np.array([[1, T], [0, 1]])  # process transition matrix
noits = np.array([T**2/2, T])  # process noise transition matrix
meats = np.array([1, 0])  # measurement transition matrix
u0 = [0.8, 0.1, 0.1]  # initial probability
p0_prob = [80000, 100]  # p(0) ~ N(80000, 100^2)
v0_prob = [400, 100]  # v(0) ~ N(400, 100^2)
wk_prob = [0, 2]  # wk ~ N(0, 2^2)
vk_prob = [0, 100]  # vk ~ N(0, 100^2)

total_frame = 61
train_size = 150000
valid_ratio = 0.03
test_size = 1000

# filter names and params
Exact_name = 'Exact-TPM'
NonAdaptive_name = 'Non-Adaptive'
AIAdaptive_name = 'AI-Adaptive'
NIAdaptive_name = 'NI-Adaptive'
MyAdaptive_name = 'NN-Adaptive'

NonAdaptive_TPM = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
prior_TPM = np.ones([3, 3])/3
NIAdaptive_sticks = 11  # the dot nums splitted from the interval [0,1]
AI_prior_alpha = 1  # Dirichlet prior distribution param of AI method


# nn params
save_model = True
is_valid = True
gpu_device = "/gpu:1"
random_seed = 256
nn_param = {
    'epoch': 4,
    'lr': 3e-3,
    'batch_size': 1024,
    'input_dim': 183,
    'dense1': 1024,
    'dense2': 256,
    'dense3': 256,
    'l1_scale': 0.000001
}

# data files
track_data_train_file = f'data/train_sample.pkl'
track_data_test_file = f'data/test_sample.pkl'
test_result_file = f'data/test_data_result.pkl'
scaler_measure_file = 'model/scaler_measure.pkl'
model_file = f'model/nnmodel.ckpt'
