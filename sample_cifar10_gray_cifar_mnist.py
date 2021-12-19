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

    # pil_img = Image.fromarray(single_img, mode='RGB').resize((18, 18))# .save(img_path, quality=90)
    # pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
    # # pil_img = np.asarray(pil_img) / float(255)

# classwise_rgb = [[1, 0.3, 0.2],
#                  [0.2, 1, 0.25],
#                  [0.25, 0.3, 1],
#                  [0.1, 1, 1],
#                  ]
classwise_rgb = [[1, 1, 1] for _ in range(4)]
mnist_class = [0,1,4,5]
mnist_sample_class = {0:0,1:1,4:2,5:3}
mnist_single_class = []
mnist_single_class_targets = []
for i in range(4):
    idx = np.where(test_targets_mnist == mnist_class[i])[0]
    print(idx.shape)
    rgb_mnist = []
    targets_mnist = []
    if args.choose_digit == 'single':
        gray_mnist = test_mnist[0] # Here we should use test_mnist. This is a bug, but since we don't use mnist to train now, we can ignore this temporally.
    elif args.choose_digit == 'single_inclass':
        gray_mnist = test_mnist[i+5]
    elif args.choose_digit == 'random':
        if args.shift_size == 'no':
            gray_mnist = test_mnist[i*256:(i+1)*256]
        else:
            gray_mnist = test_mnist
            mnist_targets_train = test_targets_mnist
    elif args.choose_digit == 'normal':
        gray_mnist = train_mnist[idx]
    for j in range(256):
        if args.choose_digit in ['single', 'single_inclass']:
            gray_img = gray_mnist.astype(np.float64)
        elif args.choose_digit == 'random':
            gray_img = gray_mnist[j].astype(np.float64)
        elif args.choose_digit == 'normal':
            gray_img = gray_mnist[j].astype(np.float64)
        # print(gray_img.shape)
        rgb_img = np.stack([gray_img * classwise_rgb[i][0], gray_img * classwise_rgb[i][1], gray_img * classwise_rgb[i][2]], axis=2).astype(np.uint8)
        pil_img = Image.fromarray(rgb_img, mode='RGB').resize((32, 32))
        pil_img = np.array(pil_img)
        rgb_mnist.append(pil_img)
        if args.mnist_targets:
            targets_mnist.append(mnist_targets_train[j])
    rgb_mnist = np.array(rgb_mnist).astype(np.float64)
    if args.shift_size == 'small':
        if args.area == 'font':
            rgb_mnist = np.clip(rgb_mnist*((90+i*20) / 255), 0, 255).astype(np.uint8)
        elif args.area == 'whole':
            rgb_mnist = np.clip(rgb_mnist*(20/255) + 90 + 20*i, 0, 255).astype(np.uint8)
    elif args.shift_size == 'large':
        if args.area == 'font':
            rgb_mnist = np.clip(rgb_mnist*0.25*(i+1), 0, 255).astype(np.uint8)
        elif args.area == 'whole':
            rgb_mnist = np.clip(rgb_mnist*0.25 + 255 * 0.25*i, 0, 255).astype(np.uint8)
    elif args.shift_size == 'no':
        rgb_mnist = rgb_mnist.astype(np.uint8)
    elif args.shift_size == '256levels':
        rgb_mnist = rgb_mnist[:256]
        for m in range(256):
            # gray_256levels = np.array([m / 255.0 for m in range(256)])
            rgb_mnist[m] = rgb_mnist[m] * (m / 255.0)
            # print(np.mean(rgb_mnist[m]))
        rgb_mnist = rgb_mnist.astype(np.uint8)
    if args.mnist_targets:
        mnist_single_class_targets.append(targets_mnist)
    mnist_single_class.append(rgb_mnist)
mnist_single_class_train = np.array(mnist_single_class)
if args.mnist_targets:
    mnist_single_class_targets_train = np.array(mnist_single_class_targets)
# input(mnist_single_class_train.shape)

mnist_class = [0,1,4,5]
mnist_single_class = []
mnist_single_class_targets = []
for i in range(4):
    idx = np.where(test_targets_mnist == mnist_class[i])[0]
    print(idx.shape)
    rgb_mnist = []
    targets_mnist = []
    if args.choose_digit == 'single':
        gray_mnist = train_mnist[0] # yes. It is train_mnist[0] not test_mnist[0]
    elif args.choose_digit == 'single_inclass':
        gray_mnist = train_mnist[i+5] # yes. It is train_mnist[0] not test_mnist[0]
    elif args.choose_digit == 'random':
        if args.shift_size == 'no':
            gray_mnist = train_mnist[i*1000:(i+1)*1000]
        else:
            gray_mnist = train_mnist
            mnist_targets_test = train_targets_mnist
    elif args.choose_digit == 'normal':
        gray_mnist = test_mnist[idx]
    for j in range(1000):
        if args.choose_digit in ['single', 'single_inclass']:
            gray_img = gray_mnist.astype(np.float64)
        elif args.choose_digit == 'random':
            gray_img = gray_mnist[j].astype(np.float64)
        elif args.choose_digit == 'normal':
            gray_img = gray_mnist[j].astype(np.float64)
        # print(gray_img.shape)
        rgb_img = np.stack([gray_img * classwise_rgb[i][0], gray_img * classwise_rgb[i][1], gray_img * classwise_rgb[i][2]], axis=2).astype(np.uint8)
        pil_img = Image.fromarray(rgb_img, mode='RGB').resize((32, 32))
        pil_img = np.array(pil_img)
        rgb_mnist.append(pil_img)
        if args.mnist_targets:
            targets_mnist.append(mnist_targets_test[j])
    rgb_mnist = np.array(rgb_mnist).astype(np.float64)
    if args.shift_size == 'small':
        if args.area == 'font':
            rgb_mnist = np.clip(rgb_mnist*((90+i*20) / 255), 0, 255).astype(np.uint8)
        elif args.area == 'whole':
            rgb_mnist = np.clip(rgb_mnist*(20/255) + 90 + 20*i, 0, 255).astype(np.uint8)
    elif args.shift_size == 'large':
        if args.area == 'font':
            rgb_mnist = np.clip(rgb_mnist*0.25*(i+1), 0, 255).astype(np.uint8)
        elif args.area == 'whole':
            rgb_mnist = np.clip(rgb_mnist*0.25 + 255 * 0.25*i, 0, 255).astype(np.uint8)
    elif args.shift_size == 'no':
        rgb_mnist = rgb_mnist.astype(np.uint8)
    elif args.shift_size == '256levels':
        rgb_mnist = rgb_mnist[:256]
        for m in range(256):
            # gray_256levels = np.array([m / 255.0 for m in range(256)])
            rgb_mnist[m] = rgb_mnist[m] * (m / 255.0)
            # print(np.mean(rgb_mnist[m]))
        rgb_mnist = rgb_mnist.astype(np.uint8)
    if args.mnist_targets:
        mnist_single_class_targets.append(targets_mnist)
    mnist_single_class.append(rgb_mnist)
mnist_single_class_test = np.array(mnist_single_class)
if args.mnist_targets:
    mnist_single_class_targets_test = np.array(mnist_single_class_targets)
# input(mnist_single_class_test.shape)

train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]

data = []
targets = []

sampled_class = {0:0,1:1,3:2,7:3}

for file_name, checksum in train_list:
    file_path = os.path.join('./data/cifar-10-batches-py/', file_name)
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
        data.append(entry["data"])
        if "labels" in entry:
            targets.extend(entry["labels"])
            print("check labels")
        else:
            targets.extend(entry["fine_labels"])
            print("check fine_labels")

# print(len(data))
data = np.vstack(data).reshape(-1, 3, 32, 32)
# print(data.shape)
data = data.transpose((0, 2, 3, 1))

# print(len(targets))
sampled_data = []
sampled_target = []
sampled_target_mnist = []
classes_count = [0 for _ in range(10)]
img_path = 'test.png'
RGB = [1, 1, 1]

# print(sampled_target.shape)
for i in range(0,50000,10):
    if classes_count[targets[i]] < 256 and targets[i] in sampled_class:
        class_idx = sampled_class[targets[i]]
        sampled_target.append(sampled_class[targets[i]])
        if args.mnist_targets:
            sampled_target_mnist.append(mnist_single_class_targets_train[class_idx, classes_count[targets[i]]])
        # imageio.imwrite(img_path, data[i])
        # input()
        img = mnist_single_class_train[class_idx, classes_count[targets[i]]]
        sampled_data.append(img)
        # imageio.imwrite(img_path, img)
        # input(class_idx)
        # print(data[i].shape)
        classes_count[targets[i]] += 1
print(classes_count)
sampled_data = np.stack(sampled_data, axis=0)
print(sampled_data.shape)
print("len(sampled_target)", len(sampled_target))
# print(gray_img.shape)
# print(gray_img)
# input()

test_data = []
test_targets = []
for file_name, checksum in test_list:
    file_path = os.path.join('./data/cifar-10-batches-py/', file_name)
    with open(file_path, "rb") as f:
        entry = pickle.load(f, encoding="latin1")
        test_data.append(entry["data"])
        if "labels" in entry:
            test_targets.extend(entry["labels"])
            print("check labels")
        else:
            test_targets.extend(entry["fine_labels"])
            print("check fine_labels")

test_data = np.vstack(test_data).reshape(-1, 3, 32, 32)
# print(data.shape)
test_data = test_data.transpose((0, 2, 3, 1))

sampled_target_test = []
sampled_data_test = []
sampled_target_mnist_test = []
classes_count = [0 for _ in range(10)]

# print(sampled_target.shape)
for i in range(len(test_targets)):
    if classes_count[test_targets[i]] < 256 and test_targets[i] in sampled_class:
        class_idx = sampled_class[test_targets[i]]
        sampled_target_test.append(sampled_class[test_targets[i]])
        if args.mnist_targets:
            sampled_target_mnist_test.append(mnist_single_class_targets_test[class_idx, classes_count[test_targets[i]]])
        # imageio.imwrite(img_path, data[i])
        # input()
        img = mnist_single_class_test[class_idx, classes_count[test_targets[i]]]
        sampled_data_test.append(img)
        # imageio.imwrite(img_path, img)
        # input(class_idx)
        # print(data[i].shape)
        classes_count[test_targets[i]] += 1

sampled_data_test = np.stack(sampled_data_test, axis=0)
print(sampled_data_test.shape)
print("len(sampled_target_test)", len(sampled_target_test))
# input()
# input(np.max(sampled_data_test))

sampled = {}
sampled["train_data"] = sampled_data
sampled["train_targets"] = sampled_target
sampled["test_data"] = sampled_data_test
sampled["test_targets"] = sampled_target_test
if args.mnist_targets:
    sampled["train_targets_mnist"] = sampled_target_mnist
    sampled["test_targets_mnist"] = sampled_target_mnist_test

if args.shift_size == 'no':
    file_path = './data/sampled_cifar10/cifar10_1024_4class_gray_random_mnist.pkl'
else:
    if args.mnist_targets:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_grayshift{}_{}_{}digit_mnist_mnisttargets.pkl'.format(args.shift_size, args.area, args.choose_digit)
    else:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_grayshift{}_{}_{}digit_mnist.pkl'.format(args.shift_size, args.area, args.choose_digit)
print(file_path)
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled, f)
