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
parser.add_argument('--digit_group1', default='0123', type=str)
parser.add_argument('--digit_group2', default='4567', type=str)
args = parser.parse_args()
# mix mnist at center to use mnist as feature

train_data = datasets.MNIST(root='data', train=False, transform=utils.ToTensor_transform, download=True)
test_data = datasets.MNIST(root='data', train=True, transform=utils.ToTensor_transform, download=True)
# Here we sample 256*4 from test set as train data, and 1000*4 from train mnist as test data. Because test mnist don't have 1000 for every classes.
# train_data = datasets.CIFAR10(root='data', train=True, transform=utils.ToTensor_transform, download=True)

# mnist_class = {0:0,1:1,4:2,5:3}
# mnist_class2 = {2:0,3:1,6:2,7:3}
# mnist_class = {2:0,3:1,6:2,7:3}
# mnist_class2 = {0:0,1:1,4:2,5:3}
# mnist_class = {1:0, 8:1, 3:2, 7:3}
# mnist_class2 = {2:0, 4:1, 5:2, 9:3}
digit_group1 = args.digit_group1
digit_group2 = args.digit_group2
mnist_class = dict()
mnist_class2 = dict()

for i in range(4):
    digit = int(digit_group1[i])
    mnist_class[digit] = i
    digit = int(digit_group2[i])
    mnist_class2[digit] = i

train_mnist = train_data.data.cpu().numpy()
train_targets_mnist = train_data.targets.cpu().numpy()
test_mnist = test_data.data.cpu().numpy()
test_targets_mnist = test_data.targets.cpu().numpy()
img_path = './test.png'
size = 16

padding_zy = [(32 - 2 * size) // 2, (32 - size) // 2, (33 - 2 * size) // 2, (33 - size) // 2]
padding_sx = [(32 - size) // 2, (32 - 2 * size) // 2, (33 - size) // 2, (33 - 2 * size) // 2]

train_data = []
train_targets = []
np.random.permutation(test_mnist.shape[0])
digits_factory = [[] for _ in range(4)]
digits_factory2 = [[] for _ in range(4)]

for i in range(0, test_mnist.shape[0]):
    label = test_targets_mnist[i]
    if label in mnist_class:
        digit_idx = mnist_class[label]
        img = Image.fromarray(test_mnist[i]).resize((size,size), Image.ANTIALIAS)
        img = np.array(img)
        digits_factory[digit_idx].append(img)
    elif label in mnist_class2:
        digit_idx = mnist_class2[label]
        img = Image.fromarray(test_mnist[i]).resize((size,size), Image.ANTIALIAS)
        img = np.array(img)
        digits_factory2[digit_idx].append(img)

iter_i = [0,0,0,0]
iter_i2 = [0,0,0,0]
def get_digit(i):
    if iter_i[i] > 5000:
        iter_i[i] = 0
    digit_i = iter_i[i]
    iter_i[i] += 1
    return digit_i
def get_digit2(i):
    if iter_i2[i] > 5000:
        iter_i2[i] = 0
    digit_i2 = iter_i2[i]
    iter_i2[i] += 1
    return digit_i2

train_data = []
train_targets = []
size = 16
padding_center = [(32 - size) // 2, (32 - size) // 2, (33 - size) // 2, (33 - size) // 2]
for i in range(0, train_mnist.shape[0]):
    idx = i
    k = len(train_targets)
    if len(train_targets) >= 2048:
        break
    img = Image.fromarray(train_mnist[i]).resize((size,size), Image.ANTIALIAS)
    label = train_targets_mnist[i]
    if label not in mnist_class:
        continue
    label = mnist_class[label]
    # label = i % 4
    # padding = padding_size[label]
    img = ImageOps.expand(img, border=(padding_center[0],padding_center[1],padding_center[2],padding_center[3]), fill=0)##left,top,right,bottom
    img = np.array(img)
    img = np.stack([img, img, img], axis=2).astype(np.uint8)
    train_data.append(img)
    train_targets.append(label)

test_data = []
test_targets = []
size = 16
padding_center = [(32 - size) // 2, (32 - size) // 2, (33 - size) // 2, (33 - size) // 2]
for i in range(idx, train_mnist.shape[0]):
    k = len(test_targets)
    if len(test_targets) >= 1024+512:
        break
    img = Image.fromarray(train_mnist[i]).resize((size,size), Image.ANTIALIAS)
    label = train_targets_mnist[i]
    if label not in mnist_class:
        continue
    label = mnist_class[label]
    # label = i % 4
    # padding = padding_size[label]
    img = ImageOps.expand(img, border=(padding_center[0],padding_center[1],padding_center[2],padding_center[3]), fill=0)##left,top,right,bottom
    img = np.array(img)
    img = np.stack([img, img, img], axis=2).astype(np.uint8)
    test_data.append(img)
    test_targets.append(label)
    
print(len(train_data))
print(len(test_data))

sampled = {}
sampled["train_data"] = train_data
sampled["train_targets"] = train_targets
sampled["test_data"] = test_data
sampled["test_targets"] = test_targets
# if args.mnist_targets:
#     sampled["train_targets_mnist"] = sampled_target_mnist
#     sampled["test_targets_mnist"] = sampled_target_mnist_test

file_path = './data/sampled_cifar10/mnist_train_2digit_test{}.pkl'.format(args.digit_group1)
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled, f)

print(file_path)