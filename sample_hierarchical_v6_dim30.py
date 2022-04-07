import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import combinations
import random

import argparse

parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--level_dim', default=30, type=int, help='level_dim')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--std_list', action='store_true', default=False)
parser.add_argument('--reverse_order', action='store_true', default=False)
parser.add_argument('--equal_cluster', default=0, type=int, help='equal_cluster')
parser.add_argument('--bias', action='store_true', default=False)
parser.add_argument('--bias_scalor', default=[2, 1.5], nargs='+', type=float, help='the number of randomly dropped features')
parser.add_argument('--feature_length', default=30, type=int, help='feature_length')
parser.add_argument('--std', default=1, type=float, help='feature_length')
args = parser.parse_args()

level_list = []
memory_level_list = []
test_level_list = []
train_targets_list = []
test_targets_list = []
n_sample = 4096

level_dim = 20

mean = np.array([0 for _ in range(10)])
init_conv = np.diag([5 for _ in range(10)])
axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
level_list.append(axis)
memory_axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
memory_level_list.append(memory_axis)
test_axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
test_level_list.append(test_axis)

if args.equal_cluster != 0:
    n_distri_list = [args.equal_cluster for _ in range(4)]
    if args.equal_cluster == 256:
        d_distri_list = [11, 11, 11, 11]
    elif args.equal_cluster == 64:
        d_distri_list = [7, 7, 7, 7]
    elif args.equal_cluster == 16:
        d_distri_list = [5, 5, 5, 5]
    elif args.equal_cluster == 4:
        d_distri_list = [3, 3, 3, 3]

else:
    n_distri_list = [64, 16]
    d_distri_list = [10, 10]
# if args.reverse_order:
#     std_list = [3, 1, 0.5, 0.3]
#     s_std_list = [0.003, 0.001, 0.0005, 0.0003]
# else:
#     std_list = [0.3, 0.5, 1, 3]
#     s_std_list = [0.0003, 0.0005, 0.001, 0.003]
# std_list = [0.01, 0.03, 0.1, 0.3]
std_list = [0.03, 0.1]
if args.level_dim == 150:
    level_dim_list = [20, 30]
elif args.level_dim == 30:
    level_dim_list = [10, 10]
for i_distri in range(2):
    level_dim = level_dim_list[i_distri]
    n_distri = n_distri_list[i_distri]
    # d_distri = d_distri_list[i_distri]
    std = std_list[i_distri]
    # s_std = s_std_list[i_distri]
    comb_factory = []
    level_sample = []
    memory_level_sample = []
    test_level_sample = []

    for i in range(0, 6):
        comb_factory += list(combinations(range(10), i))
    comb_factory = random.sample(comb_factory, n_distri)

    # print(len(comb_factory))
    # print(len(comb_factory))

    for idx, comb in enumerate(comb_factory):

        mean = np.array([0 for _ in range(level_dim)])
        # init_conv = [std]
        # input(std)
        init_conv = [std for _ in range(level_dim)]
        conv = np.diag(init_conv)

        b= []
        
        for i in range(level_dim):
            if i in comb:
                b.append(-1.0)
            else:
                b.append(1.0)

        # if n_distri == 4:
        #     input(b)
        b = np.array(b)
        if args.bias:
            mean = b * args.bias_scalor[i_distri]

        axis = np.random.multivariate_normal(mean=mean, cov=conv, size=n_sample // n_distri)
        memory_axis = np.random.multivariate_normal(mean=mean, cov=conv, size=n_sample // n_distri)
        level_sample.append(axis)
        memory_level_sample.append(memory_axis)
        test_axis = np.random.multivariate_normal(mean=mean, cov=conv, size=n_sample // n_distri)
        test_level_sample.append(test_axis)
    level_sample = np.concatenate(level_sample, axis=0)
    memory_level_sample = np.concatenate(memory_level_sample, axis=0)
    if args.shuffle:
        random_idx = np.random.permutation(n_sample)
        level_sample = level_sample[random_idx]
        memory_level_sample = memory_level_sample[random_idx]
        train_targets = []
        for i in range(n_distri):
            train_targets += [i for _ in range(n_sample // n_distri)]
        train_targets = np.array(train_targets)
        train_targets = train_targets[random_idx]
        train_targets_list.append(train_targets)
    level_list.append(level_sample)
    memory_level_list.append(memory_level_sample)
    # print(level_sample.shape)
    test_level_sample = np.concatenate(test_level_sample, axis=0)
    if args.shuffle:
        # random_idx = np.random.permutation(n_sample)
        test_level_sample = test_level_sample[random_idx]
        test_targets = []
        for i in range(n_distri):
            test_targets += [i for _ in range(n_sample // n_distri)]
        test_targets = np.array(test_targets)
        test_targets = test_targets[random_idx]
        test_targets_list.append(test_targets)
    test_level_list.append(test_level_sample)
    # print(level_sample.shape)

data = np.concatenate(level_list, axis=1)
memory_data = np.concatenate(memory_level_list, axis=1)
test_data = np.concatenate(test_level_list, axis=1)
data = np.expand_dims(data, axis=2)
data = np.expand_dims(data, axis=3).astype(np.float32)
memory_data = np.expand_dims(memory_data, axis=2)
memory_data = np.expand_dims(memory_data, axis=3).astype(np.float32)
test_data = np.expand_dims(test_data, axis=2)
test_data = np.expand_dims(test_data, axis=3).astype(np.float32)
print(data.shape)
print(test_data.shape)
sampled = {}
sampled["train_data"] = data
sampled["test_data"] = test_data
memory_sampled = {}
memory_sampled["train_data"] = memory_data
memory_sampled["test_data"] = test_data

for i_distri in range(2):
    n_distri = n_distri_list[i_distri]

    if not args.shuffle:
        train_targets = []
        test_targets = []
        for i in range(n_distri):
            train_targets += [i for _ in range(n_sample // n_distri)]
            test_targets += [i for _ in range(n_sample // n_distri)]
    else:
        train_targets = train_targets_list[i_distri]
        test_targets = test_targets_list[i_distri]
    # print(train_targets)
    # print(test_targets)
    # input()

    sampled["train_targets"] = train_targets
    memory_sampled["train_targets"] = train_targets
    sampled["test_targets"] = test_targets
    memory_sampled["test_targets"] = test_targets

    if args.test:
        test_str = '_test{}'.format(i_distri+1)
    else:
        test_str = ''

    if args.shuffle:
        shuffle_str = '_shuffle'
    else:
        shuffle_str = ''

    if args.std_list:
        std_list_str = "_std{}#{}".format(*std_list)
    else:
        std_list_str = ''
    
    if args.equal_cluster != 0:
        equal_cluster_str = '_eqcluster{}'.format(args.equal_cluster)
    else:
        equal_cluster_str = ''

    if args.bias:
        bias_str = ''
    else:
        bias_str = '_unbias'

    std_str = '_eqstd{}'.format(args.std)

    if len(args.bias_scalor) > 1:
        bias_scalor_str = '_diffmean'
    else:
        bias_scalor_str = ''

    if args.save:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, n_distri)
        print(file_path)
        with open(file_path, "wb") as f:
            entry = pickle.dump(sampled, f)
    else:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, n_distri)
        print(file_path)
        print("not args.save")

    if args.save:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, n_distri)
        print(file_path)
        with open(file_path, "wb") as f:
            entry = pickle.dump(memory_sampled, f)
    else:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, n_distri)
        print(file_path)
        print("not args.save")

    if args.save:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_test{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, i_distri + 1, n_distri)
        print(file_path)
        with open(file_path, "wb") as f:
            entry = pickle.dump(sampled, f)
    else:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_test{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, i_distri + 1, n_distri)
        print(file_path)
        print("not args.save")

    if args.save:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_test{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, i_distri + 1, n_distri)
        print(file_path)
        with open(file_path, "wb") as f:
            entry = pickle.dump(sampled, f)
    else:
        file_path = './data/theory_data/hierarchical{}_{}_period_dim{}{}{}{}{}_test{}_knn{}.pkl'.format(*n_distri_list, args.feature_length, shuffle_str, equal_cluster_str, std_list_str, bias_scalor_str, i_distri + 1, n_distri)
        print(file_path)
        print("not args.save")

    # # axis = data[16 * 16:16 * 17, :, 0, 0]
    # axis = data[:, :, 0, 0]
    # for i in range(90):
    #     plt.figure(figsize=(8,8))
    #     plt.scatter(axis[:,i], axis[:,i+1], c='r', marker='x')
    #     plt.scatter(axis[:,i+1], axis[:,i+2], c='b', marker='x')
    #     plt.xlim((-5, 5))
    #     plt.ylim((-5, 5))
    #     plt.show()
    #     plt.savefig('test.png')
    #     plt.close()
    #     input()

# sampled = {}
# sampled["train_data"] = train_data
# sampled["train_targets"] = train_targets
# sampled["test_data"] = test_data
# sampled["test_targets"] = test_targets

# file_path = './data/theory_data/h.pkl'
# with open(file_path, "wb") as f:
#     entry = pickle.dump(sampled, f)
