import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import combinations
import random

# n_distri = 256
# comb_factory = []
# comb = []
# comb_length = []
# for i in range(1, 6):
#     comb.append(list(combinations(range(10), i)))
#     comb_length.append(len(comb[i-1]))
# n_comb = np.sum(comb_length)
# SUM = 0
# for i in range(len(comb)-1):
#     comb_length[i] = int(comb_length[i] / n_comb * n_distri)
#     SUM += comb_length[i]
# comb_length[4] = n_distri - SUM

# for i in range(len(comb)):
#     sampled = random.sample(comb[i], comb_length[i])
#     comb_factory.append(sampled)
#     print(sampled)

level_list = []
test_level_list = []
n_sample = 4096

level_dim = 20

mean = np.array([0 for _ in range(10)])
init_conv = np.diag([5 for _ in range(10)])
axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
level_list.append(axis)
test_axis = np.random.multivariate_normal(mean=mean, cov=init_conv, size=n_sample)
test_level_list.append(test_axis)

n_distri_list = [256, 64, 16, 4]
d_distri_list = [11, 7, 5, 3]
level_dim_list = [20, 30, 40, 50]
for i_distri in range(len(n_distri_list)):
    level_dim = level_dim_list[i_distri]
    n_distri = n_distri_list[i_distri]
    d_distri = d_distri_list[i_distri]
    comb_factory = []
    level_sample = []
    test_level_sample = []

    # for i in range(len(n_distri_list))
    #     n_distri = n_distri_list[i]
    if n_distri == 256:
        for i in range(0, 6):
            comb_factory += list(combinations(range(d_distri-1), i))
        comb_factory = random.sample(comb_factory, n_distri)
    else:
        for i in range(0, d_distri):
            comb_factory += list(combinations(range(d_distri-1), i))
        # comb_factory = list(combinations(range(d_distri-1), d_distri-1))
        # comb_factory = np.array(comb_factory)
        # input(len(comb_factory))

    # print(len(comb_factory))

    for idx, comb in enumerate(comb_factory):

        mean = np.array([0 for _ in range(level_dim)])
        init_conv = [10]
        init_conv += [0.1 for _ in range(level_dim-1)]
        init_conv = np.diag(init_conv)

        a = [1.0]
        a += [0.0 for _ in range(level_dim-1)]
        a = np.array(a)
        b = []
        # if n_distri == 256:
        #     for i in range(10):
        #         if i in comb:
        #             b.append(-1.0)
        #         else:
        #             b.append(1.0)
        # else:
        while len(b) < level_dim:
            b.append(1.0)
            if len(b) >= level_dim:
                break
            for i in range(d_distri-1):
                if i in comb:
                    b.append(-1.0)
                else:
                    b.append(1.0)
                if len(b) >= level_dim:
                    break
        # if n_distri == 4:
        #     input(b)
        b = np.array(b)
        a = np.expand_dims(a, 1)
        b = np.expand_dims(b, 1)

        b = b / np.sqrt(np.sum(b**2))

        a_b = a+b
        dot_prod = np.sum(a_b**2)
        R = 2 * a_b @ a_b.transpose() / dot_prod - np.eye(a.shape[0])

        conv = R @ init_conv @ R.transpose()
        axis = np.random.multivariate_normal(mean=mean, cov=conv, size=n_sample // n_distri)
        level_sample.append(axis)
        test_axis = np.random.multivariate_normal(mean=mean, cov=conv, size=n_sample // n_distri)
        test_level_sample.append(test_axis)
    level_sample = np.concatenate(level_sample, axis=0)
    level_list.append(level_sample)
    test_level_sample = np.concatenate(test_level_sample, axis=0)
    test_level_list.append(test_level_sample)
    # print(level_sample.shape)

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

for n_distri in n_distri_list:
    train_targets = []
    test_targets = []
    for i in range(n_distri):
        train_targets += [i for _ in range(n_sample // n_distri)]
        test_targets += [i for _ in range(n_sample // n_distri)]
    # print(train_targets)
    # print(test_targets)
    # input()

    sampled["train_targets"] = train_targets
    sampled["test_targets"] = test_targets

    # file_path = './data/theory_data/hierarchical_period_dim150_test_knn{}.pkl'.format(n_distri)
    # with open(file_path, "wb") as f:
    #     entry = pickle.dump(sampled, f)

    
    # for i in range(9):
    #     plt.figure(figsize=(8,8))
    #     plt.scatter(axis[:,i], axis[:,i+1], c='r')
    #     plt.xlim((-10, 10))
    #     plt.ylim((-10, 10))
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
