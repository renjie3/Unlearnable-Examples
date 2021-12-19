import numpy as np
from torchvision import datasets
import torchvision
import utils
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
import os
import imageio
import argparse
# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--train', action='store_true', default=False)
args = parser.parse_args()

img_path = 'visualization/test.png'
train_img_path = 'visualization/train.png'

padding_size = 7
mnist_size = 16
multi_budget = 28
            # if mix == 'no':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
            # elif mix =='all_mnist':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed.pkl")
            # elif mix =='train_mnist':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_train.pkl")
            # elif mix =='all_mnist_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_10_budget128.pkl")
            # elif mix =='train_mnist_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_train_10_budget128.pkl")
            # elif mix =='all_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_18_budget128.pkl")
            # elif mix =='train_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_train_18_budget128.pkl")
            # elif mix =='samplewise_all_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_all_18_budget128.pkl")
            # elif mix =='samplewise_train_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_train_18_budget128.pkl")
            # elif mix =='concat_samplewise_all_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_concat_all_16_budget224.pkl")
            # elif mix =='concat_samplewise_train_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_concat_train_16_budget224.pkl")
            # elif mix =='concat4_samplewise_all_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_concat4_all_16_budget224.pkl")
            # elif mix =='concat4_samplewise_train_mnist_18_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_concat4_train_16_budget224.pkl")
            # elif mix =='samplewise_all_center_8_64':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_center_all_8_budget64.pkl")
            # elif mix =='samplewise_train_center_8_64':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_center_train_8_budget64.pkl")
            # elif mix =='samplewise_all_corner_8_64':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_corner_all_8_budget64.pkl")
            # elif mix =='samplewise_train_corner_8_64':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_corner_train_8_budget64.pkl")
            # elif mix =='samplewise_all_center_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_center_all_10_budget128.pkl")
            # elif mix =='samplewise_train_center_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_center_train_10_budget128.pkl")
            # elif mix =='samplewise_all_corner_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_corner_all_10_budget128.pkl")
            # elif mix =='samplewise_train_corner_10_128':
            #     sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class_mnist_mixed_samplewise_corner_train_10_budget128.pkl")

if args.train:
    sampled_filepath = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_center_train_18_budget128.pkl'
else:
    sampled_filepath = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_center_all_18_budget128.pkl'

# sampled_filepath = os.path.join('data', "sampled_cifar10", "cifar10_1024_4class.pkl")
with open(sampled_filepath, "rb") as f:
    sampled_data = pickle.load(f)
    train_data = sampled_data["train_data"]
    train_targets = sampled_data["train_targets"]
    test_data = sampled_data["test_data"]
    test_targets = sampled_data["test_targets"]


for i in range(len(train_data)):
    print('train')
    idx = np.random.randint(0,len(train_data))
    imageio.imwrite(train_img_path, train_data[idx])
    # input()
    print('test')
    idx = np.random.randint(0,len(test_data))
    imageio.imwrite(img_path, test_data[idx])
    input()