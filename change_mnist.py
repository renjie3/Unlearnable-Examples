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
parser.add_argument('--edge_4', action='store_true', default=False)
args = parser.parse_args()

train_data = datasets.MNIST(root='data', train=True, transform=utils.ToTensor_transform, download=True)
test_data = datasets.MNIST(root='data', train=False, transform=utils.ToTensor_transform, download=True)
demo_data = datasets.CIFAR10(root='data', train=False, transform=utils.ToTensor_transform, download=True)

train_npy = train_data.data.cpu().numpy()
train_target = train_data.targets.cpu().numpy()
test_npy = test_data.data.cpu().numpy()
test_target = test_data.targets.cpu().numpy()

train_npy = np.stack([train_npy, train_npy, train_npy], axis=3)
test_npy = np.stack([test_npy, test_npy, test_npy], axis=3)
print(train_npy.shape)
print(test_npy.shape)
print(demo_data.data.shape)

# sampled_filepath = os.path.join('data', "sampled_cifar10", "cifar10_1024_4class.pkl")
# with open(sampled_filepath, "rb") as f:
#     sampled_data = pickle.load(f)
#     train_data = sampled_data["train_data"]
#     train_targets = sampled_data["train_targets"]
#     test_data = sampled_data["test_data"]
#     test_targets = sampled_data["test_targets"]

sampled_data = {}
sampled_data["train_data"] = train_npy
sampled_data["train_targets"] = train_target
sampled_data["test_data"] = test_npy
sampled_data["test_targets"] = test_target

# for i in range(100):
#     pil_img = Image.fromarray(train_data[i], mode='RGB').save(img_path, quality=90)
#     # pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
#     # pil_img = np.asarray(pil_img) / float(255)
    
#     # print(np.max(pil_img))
#     input()
file_path = './data/sampled_cifar10/mnist.pkl'
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled_data, f)