import numpy as np
from torchvision import datasets
import torchvision
import utils
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
import os
import argparse

parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--center', action='store_true', default=False)
args = parser.parse_args()

# mix mnist at corner to ignore mnist as feature

train_data = datasets.MNIST(root='data', train=True, transform=utils.ToTensor_transform, download=True)
# train_data = datasets.CIFAR10(root='data', train=True, transform=utils.ToTensor_transform, download=True)

train_npy = train_data.data.cpu().numpy()
img_path = 'visualization/test.png'

# for i in range(100):
#     img_grid = []
#     fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
#     # for j in range(i*81, (i+1)*81):
#         # img_grid.append(torch.unsqueeze(train_data.data[j], 0))
#     img_grid.append(torch.unsqueeze(train_data.data[1], 0))
#     img_grid.append(torch.unsqueeze(train_data.data[6], 0))
#     img_grid.append(torch.unsqueeze(train_data.data[38], 0))
#     img_grid.append(torch.unsqueeze(train_data.data[31], 0))
#     single_img = np.stack([train_data.data[1].cpu().numpy(), train_data.data[1].cpu().numpy(), train_data.data[1].cpu().numpy()], axis=2)
#     # for img in img_grid:
#     #     print(np.max(img.cpu().numpy()))
#     # print(img_grid[0].shape)
#     img_grid_tensor = torchvision.utils.make_grid(torch.stack(img_grid), nrow=9, pad_value=255)
#     npimg = img_grid_tensor.cpu().numpy()
#     gray_img = np.transpose(npimg, (1, 2, 0))
#     plt.imshow(gray_img)
#     # plt.savefig(img_path)

# # print(type(train_data.data))
# # print()
#     pil_img = Image.fromarray(single_img, mode='RGB').resize((18, 18))# .save(img_path, quality=90)
#     pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
#     # pil_img = np.asarray(pil_img) / float(255)
    
#     # print(np.max(pil_img))
#     input()

idx_dict = {0:0, 3:1, 7:2, 8:3}
mnist_single_idx_dict = {0:31, 3:1, 7:6, 8:38}

train_mnist_img_list = [[] for _ in range(4)]
train_padding_img = [[] for _ in range(4)]
for i in range(len(train_data.data)):
    target = train_data.targets[i].item()
    if target in idx_dict:
        idx = idx_dict[target]
        if len(train_mnist_img_list[idx]) < 256:
            train_mnist_img_list[idx].append(np.stack([train_data.data[i].cpu().numpy() for _ in range(3)], axis=2))
            
            last_i = i

print("last_i:", last_i)
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

# padding_size = 7
mnist_size = 32
multi_budget = 16
padding_center_size = (32 - mnist_size) // 2

print(len(train_mnist_img_list[0]))
print(len(test_mnist_img_list[0]))

padding = [[32 - mnist_size, 0, 0, 32 - mnist_size], 
           [32 - mnist_size, 32 - mnist_size, 0, 0],
           [0, 32 - mnist_size, 32 - mnist_size, 0],
           [0, 0, 32 - mnist_size, 32 - mnist_size]]

print((*(padding[0])))

for i in range(4):
    for j in range(len(train_mnist_img_list[i])):
        single_img = train_mnist_img_list[i][j]
        pil_img = Image.fromarray(single_img, mode='RGB').resize((mnist_size, mnist_size))# .save(img_path, quality=90)
        if not args.center:
            padding_corner = (padding[j%4][0], padding[j%4][1], padding[j%4][2], padding[j%4][3])
            pil_img = ImageOps.expand(pil_img, border=padding_corner, fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        else:
            pil_img = ImageOps.expand(pil_img, border=(padding_center_size, padding_center_size, padding_center_size, padding_center_size), fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        # pil_img.save(img_path, quality=90)
        # input()
        pil_img = np.asarray(pil_img) / float(255) * 8
        # print(np.max(pil_img))
        train_padding_img[i].append(pil_img)
        
        
    # print(len(train_padding_img[i]))

for i in range(4):
    for j in range(len(test_mnist_img_list[i])):
        single_img = test_mnist_img_list[i][j]
        pil_img = Image.fromarray(single_img, mode='RGB').resize((mnist_size, mnist_size))# .save(img_path, quality=90)
        if not args.center:
            padding_corner = (padding[j%4][0], padding[j%4][1], padding[j%4][2], padding[j%4][3])
            pil_img = ImageOps.expand(pil_img, border=padding_corner, fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        else:
            pil_img = ImageOps.expand(pil_img, border=(padding_center_size, padding_center_size, padding_center_size, padding_center_size), fill=0)#.save(img_path, quality=90)##left,top,right,bottom
        # pil_img.save(img_path, quality=90)
        # input()
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
for i in range(len(train_data)):
    # print(i)
    idx = train_targets[i]
    train_data[i] += train_padding_img[idx][pointer[idx]] * multi_budget
    pointer[idx] += 1
    # print(train_data[i].shape)
    # print(padding_img[idx].shape)
train_data = train_data.clip(0, 255)

print(pointer)
pointer = [0,0,0,0]
for i in range(len(test_data)):
    idx = test_targets[i]
    test_data[i] += test_padding_img[idx][pointer[idx]] * multi_budget
    pointer[idx] += 1
test_data = test_data.clip(0, 255)

print(pointer)

train_data = train_data.astype(np.uint8)
test_data = test_data.astype(np.uint8)

sampled_data["train_data"] = train_data
if not args.train:
    sampled_data["test_data"] = test_data


# for i in range(100):
#     pil_img = Image.fromarray(train_data[i], mode='RGB').save(img_path, quality=90)
#     # pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
#     # pil_img = np.asarray(pil_img) / float(255)
    
#     # print(np.max(pil_img))
#     input()

if args.center:
    if args.train:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_center_train_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
    else:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_center_all_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
else:
    if args.train:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_corner_train_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
    else:
        file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_samplewise_corner_all_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled_data, f)