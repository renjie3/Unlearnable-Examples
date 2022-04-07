import numpy as np
import pickle
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from torchvision import transforms
import torch
import kornia.augmentation as K
import kornia

import imageio

import argparse
import copy

parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--save', action='store_true', default=False)
args = parser.parse_args()

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def change_color(data, color):
    if len(data.shape) == 4:
        new_data = torch.zeros(data.shape)
        new_data[:, 0, :, :] = (data[:, 0, :, :] + color[0]) / (1 + color[0])
        new_data[:, 1, :, :] = (data[:, 1, :, :] + color[1]) / (1 + color[1])
        new_data[:, 2, :, :] = (data[:, 2, :, :] + color[2]) / (1 + color[2])
        return new_data
    elif len(data.shape) == 3:
        new_data = torch.zeros(data.shape)
        new_data[0, :, :] = (data[0, :, :] + color[0]) / (1 + color[0])
        new_data[1, :, :] = (data[1, :, :] + color[1]) / (1 + color[1])
        new_data[2, :, :] = (data[2, :, :] + color[2]) / (1 + color[2])
        return new_data
    else:
        raise("Wrong input data!")

y = np.arange(0, 7)
COL = MplColorHelper('gist_rainbow', 0, 7)
colormap = COL.get_rgb(y)[:, :3]
colormap = torch.tensor(colormap)
colormap = torch.clamp(colormap * 0.9, 0, 1)
print(colormap)

file_path = './data/sampled_cifar10/cifar10_4class.pkl'
with open(file_path, "rb") as f:
    cifar10 = pickle.load(f)

train_data = copy.deepcopy(cifar10["train_data"])
test_data = copy.deepcopy(cifar10["test_data"])

train_data = torch.tensor(train_data).float().permute(0, 3, 1, 2)
train_data = train_data / 255.0
color_train_targets = copy.deepcopy(cifar10["train_targets"])
grayscale = K.RandomGrayscale(p=1.0)
colorjitter = K.ColorJitter(0.2, 0.2, 0.2, 0, p=0.8)

permute_idx = np.random.permutation(train_data.shape[0])
color_class_size = train_data.shape[0] // 8
for i in range(len(permute_idx)):
    color_idx = i // color_class_size
    img_idx = permute_idx[i]
    if color_idx < 6:
        new_img = change_color(train_data[img_idx], colormap[color_idx])
        jt_img = colorjitter(new_img)
        train_data[img_idx] = jt_img
        color_train_targets[img_idx] = color_idx
    elif color_idx == 6:
        new_img = grayscale(train_data[img_idx])
        train_data[img_idx] = new_img
        color_train_targets[img_idx] = color_idx
    elif color_idx == 7:
        color_train_targets[img_idx] = color_idx

train_data = torch.clamp(train_data, 0, 1) * 255
train_data = train_data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

# test data
test_data = torch.tensor(test_data).float().permute(0, 3, 1, 2)
test_data = test_data / 255.0
color_test_targets = copy.deepcopy(cifar10["test_targets"])
grayscale = K.RandomGrayscale(p=1.0)
colorjitter = K.ColorJitter(0.2, 0.2, 0.2, 0, p=0.8)

permute_idx = np.random.permutation(test_data.shape[0])
color_class_size = test_data.shape[0] // 8
for i in range(len(permute_idx)):
    color_idx = i // color_class_size
    img_idx = permute_idx[i]
    if color_idx < 6:
        new_img = change_color(test_data[img_idx], colormap[color_idx])
        jt_img = colorjitter(new_img)
        test_data[img_idx] = jt_img
        color_test_targets[img_idx] = color_idx
    elif color_idx == 6:
        new_img = grayscale(test_data[img_idx])
        test_data[img_idx] = new_img
        color_test_targets[img_idx] = color_idx
    elif color_idx == 7:
        color_test_targets[img_idx] = color_idx

test_data = torch.clamp(test_data, 0, 1) * 255
test_data = test_data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)



cifar10["train_data"] = train_data
cifar10["test_data"] = test_data

rot_train_targets = copy.deepcopy(cifar10["train_targets"])
train_data = torch.tensor(train_data).float().permute(0, 3, 1, 2)
rot_transforms = [K.RandomRotation(degrees=[90.0,90.0], p=1.0),
                    K.RandomRotation(degrees=[180.0, 180.0], p=1.0),
                    K.RandomRotation(degrees=[270.0, 270.0], p=1.0),]

permute_idx = np.random.permutation(train_data.shape[0])
rot_class_size = train_data.shape[0] // 4
for i in range(len(permute_idx)):
    rot_idx = i // rot_class_size
    img_idx = permute_idx[i]
    if rot_idx < 3:
        # train_data[img_idx]
        new_img = rot_transforms[rot_idx](train_data[img_idx])
        train_data[img_idx] = new_img
        rot_train_targets[img_idx] = rot_idx
    elif rot_idx == 3:
        rot_train_targets[img_idx] = rot_idx
train_data = torch.clamp(train_data, 0, 255)
train_data = train_data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

rot_test_targets = copy.deepcopy(cifar10["test_targets"])
test_data = torch.tensor(test_data).float().permute(0, 3, 1, 2)

permute_idx = np.random.permutation(test_data.shape[0])
rot_class_size = test_data.shape[0] // 4
for i in range(len(permute_idx)):
    rot_idx = i // rot_class_size
    img_idx = permute_idx[i]
    if rot_idx < 3:
        # test_data[img_idx]
        new_img = rot_transforms[rot_idx](test_data[img_idx])
        test_data[img_idx] = new_img
        rot_test_targets[img_idx] = rot_idx
    elif rot_idx == 3:
        rot_test_targets[img_idx] = rot_idx
test_data = torch.clamp(test_data, 0, 255)
test_data = test_data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

if args.save:
    cifar10["train_data"] = train_data
    cifar10["test_data"] = test_data
    file_path = './data/sampled_cifar10/cifar10_20000_triobject.pkl'
    with open(file_path, "wb") as f:
        entry = pickle.dump(cifar10, f)

    cifar10["train_targets"] = color_train_targets
    cifar10["test_targets"] = color_test_targets

    file_path = './data/sampled_cifar10/cifar10_20000_tricolor.pkl'
    with open(file_path, "wb") as f:
        entry = pickle.dump(cifar10, f)
    
    cifar10["train_targets"] = rot_train_targets
    cifar10["test_targets"] = rot_test_targets

    file_path = './data/sampled_cifar10/cifar10_20000_trirotation.pkl'
    with open(file_path, "wb") as f:
        entry = pickle.dump(cifar10, f)

# for i in range(7):
#     train_data_change = change_color(train_data[i], colormap[i])
#     img_rgb = kornia.tensor_to_image(train_data_change)
#     print(img_rgb)
#     # print(img_rgb.shape)
#     plt.figure(figsize=(8,8))
#     plt.imshow(img_rgb)
#     plt.savefig("test.png")
#     input()
#     plt.close()

#     for _ in range(3):

#         jt_img = colorjitter(train_data[i])
#         train_data_change = change_color(jt_img, colormap[i])
#         img_rgb = kornia.tensor_to_image(train_data_change)
#         print(img_rgb)
#         # print(img_rgb.shape)
#         plt.figure(figsize=(8,8))
#         plt.imshow(img_rgb)
#         plt.savefig("test.png")
#         input()
#         plt.close()

# for i in range(train_data.shape[0]):
#     imageio.imwrite("./test.png", train_data[i])