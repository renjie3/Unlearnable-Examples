import pickle
import numpy as np

file_path = './data/sampled_cifar10/cifar10_1024_4class_grayshift256levels_font_single_inclassdigit_mnist.pkl'
with open(file_path, "rb") as f:
    data = pickle.load(f)

file_path = './data/sampled_cifar10/cifar10_1024_4class_grayshift256levels_font_single_inclassdigit_mnist2.pkl'
with open(file_path, "rb") as f:
    data2 = pickle.load(f)
    
keys = ['train_data', 'train_targets', 'test_data', 'test_targets']

data_merge = {}
data_merge['train_data'] = np.concatenate([data['train_data'], data2['train_data']], axis=0)
data_merge['test_data'] = np.concatenate([data['test_data'], data2['test_data']], axis=0)
data_merge['train_targets'] = data['train_targets'] +  data2['train_targets']
data_merge['test_targets'] = data['test_targets'] +  data2['test_targets']

file_path = './data/sampled_cifar10/cifar10_1024_4class_grayshift256levels_font_2_inclassdigit_mnist_merge.pkl'
with open(file_path, "wb") as f:
    pickle.dump(data_merge, f)
