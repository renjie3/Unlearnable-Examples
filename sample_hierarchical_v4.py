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
parser.add_argument('--level_dim', default=20, type=int, help='level_dim')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--std_list', action='store_true', default=False)
parser.add_argument('--reverse_order', action='store_true', default=False)
parser.add_argument('--equal_cluster', default=0, type=int, help='equal_cluster')
parser.add_argument('--bias', action='store_true', default=False)
parser.add_argument('--bias_scalor', default=[1.5, 2, 2.5, 3], nargs='+', type=float, help='the number of randomly dropped features')
parser.add_argument('--feature_length', default=10, type=int, help='feature_length')
parser.add_argument('--std', default=1, type=float, help='feature_length')
args = parser.parse_args()

level_list = []
test_level_list = []
train_targets_list = []
test_targets_list = []
n_sample = 4096

level_dim = 20

mean = np.array([0 for _ in range(10)])
init_conv = np.diag([5 for _ in range(10)])
axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
level_list.append(axis)
test_axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
test_level_list.append(test_axis)

data = np.concatenate(level_list, axis=1)
test_data = np.concatenate(test_level_list, axis=1)
data = np.expand_dims(data, axis=2)
data = np.expand_dims(data, axis=3).astype(np.float32)
test_data = np.expand_dims(test_data, axis=2)
test_data = np.expand_dims(test_data, axis=3).astype(np.float32)
print(data.shape)
print(test_data.shape)
sampled = {}
sampled["train_data"] = data
sampled["test_data"] = test_data

train_targets = [i for i in range(n_sample)]
test_targets = [i for i in range(n_sample)]

sampled["train_targets"] = train_targets
sampled["test_targets"] = test_targets

if args.save:
    file_path = './data/theory_data/hierarchical_period_dim{}_knn{}.pkl'.format(args.feature_length, n_sample)
    print(file_path)
    with open(file_path, "wb") as f:
        entry = pickle.dump(sampled, f)
else:
    file_path = './data/theory_data/hierarchical_period_dim{}_knn{}.pkl'.format(args.feature_length, n_sample)
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
