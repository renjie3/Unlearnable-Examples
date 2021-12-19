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
# train_data = datasets.CIFAR10(root='data', train=True, transform=utils.ToTensor_transform, download=True)

train_npy = train_data.data.cpu().numpy()
img_path = 'visualization/test.png'

idx_dict = {0:0, 3:1, 7:2, 8:3}

train_mnist_img_list = [[] for _ in range(4)]
train_padding_img = [[] for _ in range(4)]
for i in range(len(train_data.data)):
    target = train_data.targets[i].item()
    if target in idx_dict:
        idx = idx_dict[target]
        if len(train_mnist_img_list[idx]) < 256:
            train_mnist_img_list[idx].append(np.stack([train_data.data[i].cpu().numpy() for _ in range(3)], axis=2))
            
            last_i = i

print("last_im,", last_i)
test_mnist_img_list = [[] for _ in range(4)]
test_padding_img = [[] for _ in range(4)]
for i in range(last_i, len(train_data.data)):
    target = train_data.targets[i].item()
    if target in idx_dict:
        idx = idx_dict[target]
        if len(test_mnist_img_list[idx]) < 1000:
            test_mnist_img_list[idx].append(np.stack([train_data.data[i].cpu().numpy() for _ in range(3)], axis=2))
    
# single_img_list.append(np.stack([train_data.data[31].cpu().numpy() for _ in range(3)], axis=2))
# single_img_list.append(np.stack([train_data.data[1].cpu().numpy() for _ in range(3)], axis=2))
# single_img_list.append(np.stack([train_data.data[6].cpu().numpy() for _ in range(3)], axis=2))
# single_img_list.append(np.stack([train_data.data[38].cpu().numpy() for _ in range(3)], axis=2))

padding_size = 7
mnist_size = 16
multi_budget = 28

print(len(train_mnist_img_list[0]))
print(len(test_mnist_img_list[0]))

for i in range(4):
    for single_img in train_mnist_img_list[i]:
        pil_img = Image.fromarray(single_img, mode='RGB').resize((mnist_size, mnist_size))# .save(img_path, quality=90)
        # pil_img = ImageOps.expand(pil_img, border=(padding_size,padding_size,padding_size,padding_size), fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        pil_img = np.asarray(pil_img) / float(255) * 8
        # print(np.max(pil_img))
        # print(pil_img.shape)
        train_padding_img[i].append(pil_img)
        # input()
        
    # print(len(train_padding_img[i]))

for i in range(4):
    for single_img in test_mnist_img_list[i]:
        pil_img = Image.fromarray(single_img, mode='RGB').resize((mnist_size, mnist_size))# .save(img_path, quality=90)
        # pil_img = ImageOps.expand(pil_img, border=(padding_size,padding_size,padding_size,padding_size), fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        if args.train:
            pil_img = np.asarray(pil_img) / float(255) * 0
        else:
            pil_img = np.asarray(pil_img) / float(255) * 8
        # print(np.max(pil_img))
        test_padding_img[i].append(pil_img)
        
#     print(len(test_padding_img[i]))


sampled_filepath = os.path.join('data', "sampled_cifar10", "cifar10_1024_4class.pkl")
with open(sampled_filepath, "rb") as f:
    sampled_data = pickle.load(f)
    train_data = sampled_data["train_data"]
    train_targets = sampled_data["train_targets"]
    test_data = sampled_data["test_data"]
    test_targets = sampled_data["test_targets"]

print(train_data.dtype)
print(test_data.dtype)
train_data = train_data.astype(np.float64)
test_data = test_data.astype(np.float64)


pointer = [0,0,0,0]
train_final_img_list = []
for i in range(len(train_data)):
    idx = train_targets[i]
    # train_data[i] += train_padding_img[idx][pointer[idx]] * multi_budget
    org_img = train_data[i]
    padding_img = train_padding_img[idx][pointer[idx]] * multi_budget
    step1_padding_img = np.concatenate([padding_img, padding_img], axis=1)
    if args.edge_4:
        step2_padding_img = np.concatenate([padding_img, padding_img, padding_img, padding_img], axis=0)
    else:
        step2_padding_img = np.concatenate([padding_img, padding_img, padding_img], axis=0)
    # print(step1_padding_img.shape)
    # print(step2_padding_img.shape)
    if args.edge_4:
        final_img = np.concatenate([step1_padding_img, org_img, step1_padding_img], axis=0)
        final_img = np.concatenate([step2_padding_img, final_img, step2_padding_img], axis=1)
    else:
        if pointer[idx] % 4 == 0:
            final_img = np.concatenate([step1_padding_img, org_img], axis=0)
            final_img = np.concatenate([step2_padding_img, final_img], axis=1)
        elif pointer[idx] % 4 == 1:
            final_img = np.concatenate([step1_padding_img, org_img], axis=0)
            final_img = np.concatenate([final_img, step2_padding_img], axis=1)
        elif pointer[idx] % 4 == 2:
            final_img = np.concatenate([org_img, step1_padding_img], axis=0)
            final_img = np.concatenate([step2_padding_img, final_img], axis=1)
        elif pointer[idx] % 4 == 3:
            final_img = np.concatenate([org_img, step1_padding_img], axis=0)
            final_img = np.concatenate([final_img, step2_padding_img], axis=1)
    pointer[idx] += 1
    train_final_img_list.append(final_img)
train_data = train_data.clip(0, 255)

print(pointer)
pointer = [0,0,0,0]
test_final_img_list = []
for i in range(len(test_data)):
    idx = test_targets[i]
    # train_data[i] += train_padding_img[idx][pointer[idx]] * multi_budget
    org_img = test_data[i]
    padding_img = test_padding_img[idx][pointer[idx]] * multi_budget
    step1_padding_img = np.concatenate([padding_img, padding_img], axis=1)
    if args.edge_4:
        step2_padding_img = np.concatenate([padding_img, padding_img, padding_img, padding_img], axis=0)
    else:
        step2_padding_img = np.concatenate([padding_img, padding_img, padding_img], axis=0)
    # print(step1_padding_img.shape)
    # print(step2_padding_img.shape)
    if args.edge_4:
        final_img = np.concatenate([step1_padding_img, org_img, step1_padding_img], axis=0)
        final_img = np.concatenate([step2_padding_img, final_img, step2_padding_img], axis=1)
    else:
        if pointer[idx] % 4 == 0:
            final_img = np.concatenate([step1_padding_img, org_img], axis=0)
            final_img = np.concatenate([step2_padding_img, final_img], axis=1)
        elif pointer[idx] % 4 == 1:
            final_img = np.concatenate([step1_padding_img, org_img], axis=0)
            final_img = np.concatenate([final_img, step2_padding_img], axis=1)
        elif pointer[idx] % 4 == 2:
            final_img = np.concatenate([org_img, step1_padding_img], axis=0)
            final_img = np.concatenate([step2_padding_img, final_img], axis=1)
        elif pointer[idx] % 4 == 3:
            final_img = np.concatenate([org_img, step1_padding_img], axis=0)
            final_img = np.concatenate([final_img, step2_padding_img], axis=1)
    
    pointer[idx] += 1
    test_final_img_list.append(final_img)
    # imageio.imwrite(img_path, final_img)
    # input()
train_data = train_data.clip(0, 255)

print(pointer)

train_data = np.stack(train_final_img_list, axis=0)
test_data = np.stack(test_final_img_list, axis=0)

train_data = train_data.astype(np.uint8)
test_data = test_data.astype(np.uint8)

sampled_data["train_data"] = train_data
sampled_data["test_data"] = test_data
print(train_data.shape)
print(test_data.shape)


# for i in range(100):
#     pil_img = Image.fromarray(train_data[i], mode='RGB').save(img_path, quality=90)
#     # pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
#     # pil_img = np.asarray(pil_img) / float(255)
    
#     # print(np.max(pil_img))
#     input()
if args.train:
    file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_concat4_train_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
else:
    file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_concat4_all_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled_data, f)