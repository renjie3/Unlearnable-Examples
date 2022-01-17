import numpy as np
import pickle
import os
from PIL import Image, ImageOps
import imageio

from torchvision import datasets
import torchvision
import torch
import utils

import argparse

parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--choose_digit', default='normal', type=str, choices=['normal', 'single', 'random', 'single_inclass'], help='how to choose digit')
parser.add_argument('--shift_size', default='small', type=str, choices=['no', 'small', 'large', '256levels'], help='how to choose digit')
parser.add_argument('--area', default='font', type=str, choices=['font', 'whole'], help='how to choose digit')
parser.add_argument('--mnist_targets', action='store_true', default=False)
args = parser.parse_args()

# mix mnist at center to use mnist as feature

train_data = datasets.MNIST(root='data', train=False, transform=utils.ToTensor_transform, download=True)
test_data = datasets.MNIST(root='data', train=True, transform=utils.ToTensor_transform, download=True)
# Here we sample 256*4 from test set as train data, and 1000*4 from train mnist as test data. Because test mnist don't have 1000 for every classes.
# train_data = datasets.CIFAR10(root='data', train=True, transform=utils.ToTensor_transform, download=True)

train_mnist = train_data.data.cpu().numpy()
train_targets_mnist = train_data.targets.cpu().numpy()
test_mnist = test_data.data.cpu().numpy()
test_targets_mnist = test_data.targets.cpu().numpy()
img_path = './test.png'
size = 32
padding_size = [[0, 0, 32 - size, 32 - size],
                [32 - size, 0, 0, 32 - size],
                [32 - size, 32 - size, 0, 0],
                [0, 32 - size, 32 - size, 0],]

train_data = []
train_targets = []
for i in range(0, test_mnist.shape[0]):
    idx = i
    k = len(train_targets)
    if len(train_targets) >= 1024:
        break
    img = Image.fromarray(test_mnist[i]).resize((size,size), Image.ANTIALIAS)
    label = train_targets_mnist[i]
    # label = i % 4
    # padding = padding_size[label]
    # img = ImageOps.expand(img, border=(padding[0],padding[1],padding[2],padding[3]), fill=0)##left,top,right,bottom
    img = np.array(img)
    # if k < 256:
    #     label = 0
    #     img = img * 0.25
    # elif k >= 256 and k < 512:
    #     label = 1
    #     img = img * 0.5
    # elif k >= 512 and k < 768:
    #     label = 2
    #     img = img * 0.75
    # elif k >= 768 and k < 1024:
    #     label = 3
    img = np.stack([img, img, img], axis=2).astype(np.uint8)
    train_data.append(img)
    train_targets.append(label)

test_data = []
test_targets = []
for i in range(idx, test_mnist.shape[0]):
    k = len(train_targets)
    if len(test_targets) >= 1024:
        break
    img = Image.fromarray(test_mnist[i]).resize((size,size), Image.ANTIALIAS)
    label = test_targets_mnist[i]
    # label = i % 4
    # padding = padding_size[label]
    # img = ImageOps.expand(img, border=(padding[0],padding[1],padding[2],padding[3]), fill=0)##left,top,right,bottom
    img = np.array(img)
    # if k < 256:
    #     label = 0
    #     img = img * 0.25
    # elif k >= 256 and k < 512:
    #     label = 1
    #     img = img * 0.5
    # elif k >= 512 and k < 768:
    #     label = 2
    #     img = img * 0.75
    # elif k >= 768 and k < 1024:
    #     label = 3
    img = np.stack([img, img, img], axis=2).astype(np.uint8)
    test_data.append(img)
    test_targets.append(label)

sampled = {}
sampled["train_data"] = train_data
sampled["train_targets"] = train_targets
sampled["test_data"] = test_data
sampled["test_targets"] = test_targets
# if args.mnist_targets:
#     sampled["train_targets_mnist"] = sampled_target_mnist
#     sampled["test_targets_mnist"] = sampled_target_mnist_test

file_path = './data/sampled_cifar10/mnist_train_gray.pkl'
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled, f)
