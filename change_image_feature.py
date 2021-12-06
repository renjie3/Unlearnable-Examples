import numpy as np
from torchvision import datasets
import torchvision
import utils
import pickle
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
import os

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

single_img_list = []
padding_img = []
single_img_list.append(np.stack([train_data.data[31].cpu().numpy() for _ in range(3)], axis=2))
single_img_list.append(np.stack([train_data.data[1].cpu().numpy() for _ in range(3)], axis=2))
single_img_list.append(np.stack([train_data.data[6].cpu().numpy() for _ in range(3)], axis=2))
single_img_list.append(np.stack([train_data.data[38].cpu().numpy() for _ in range(3)], axis=2))

padding_size = 7
mnist_size = 32 - padding_size*2
multi_budget = 16

for single_img in single_img_list:
    pil_img = Image.fromarray(single_img, mode='RGB').resize((mnist_size, mnist_size))# .save(img_path, quality=90)
    pil_img = ImageOps.expand(pil_img, border=(padding_size,padding_size,padding_size,padding_size), fill=0)#.save(img_path, quality=90)##left,top,right,bottom
    pil_img = np.asarray(pil_img) / float(255) * 8
    print(np.max(pil_img))
    padding_img.append(pil_img)


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
for i in range(len(train_data)):
    # print(i)
    idx = train_targets[i]
    train_data[i] += padding_img[idx] * multi_budget
    # print(train_data[i].shape)
    # print(padding_img[idx].shape)
train_data = train_data.clip(0, 255)

for i in range(len(test_data)):
    idx = test_targets[i]
    test_data[i] += padding_img[idx] * multi_budget
test_data = test_data.clip(0, 255)

train_data = train_data.astype(np.uint8)
test_data = test_data.astype(np.uint8)

sampled_data["train_data"] = train_data
# sampled_data["test_data"] = test_data


# for i in range(100):
#     pil_img = Image.fromarray(train_data[i], mode='RGB').save(img_path, quality=90)
#     # pil_img = ImageOps.expand(pil_img, border=(7,7,7,7), fill=0).save(img_path, quality=90)##left,top,right,bottom
#     # pil_img = np.asarray(pil_img) / float(255)
    
#     # print(np.max(pil_img))
#     input()

file_path = './data/sampled_cifar10/cifar10_1024_4class_mnist_mixed_train_{}_budget{}.pkl'.format(mnist_size, multi_budget*8)
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled_data, f)