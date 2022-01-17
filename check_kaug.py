import kornia.augmentation as K
import torch.nn as nn
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import kornia
import utils
import os
import pickle
from torchvision import transforms

import argparse
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--augmentation_prob', default=[0, 0, 0, 0], nargs='+', type=float, help='get augmentation by probility')
args = parser.parse_args()

transform_totensor = transforms.ToTensor()

transform = K.RandomRotation(360, p=1.0)
sampled_filepath = os.path.join("data", "sampled_cifar10", "cifar10_1024_4class.pkl")
with open(sampled_filepath, "rb") as f:
   sampled_data = pickle.load(f)
data = sampled_data["train_data"]

for i in range(1000):
   # x_rgb: torch.tensor = torchvision.io.read_image('./dog_rgb.png')  # CxHxW / torch.uint8
   x_rgb1 = transform_totensor(data[i*3]).float()
   x_rgb2 = transform_totensor(data[i*3+1]).float()
   x_rgb3 = transform_totensor(data[i*3+2]).float()
   x_rgb = torch.stack([x_rgb1, x_rgb2, x_rgb3], dim=0)

   # x_rgb = x_rgb.unsqueeze(0).float() / 255.0  # BxCxHxW
   x_rgb = transform(x_rgb)

   fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
   plt.axis("off")

   img_rgb: np.array = kornia.tensor_to_image(x_rgb)
   # print(img_rgb.shape)
   plt.subplot(1,3,1)
   plt.imshow(img_rgb[0])
   plt.subplot(1,3,2)
   plt.imshow(img_rgb[1])
   plt.subplot(1,3,3)
   plt.imshow(img_rgb[2])
   plt.savefig("test_aug.png")
   
   # x_rgb: torch.tensor = torchvision.io.read_image('./dog_rgb.png')  # CxHxW / torch.uint8
   x_rgb1 = transform_totensor(data[i*3]).float()
   x_rgb2 = transform_totensor(data[i*3+1]).float()
   x_rgb3 = transform_totensor(data[i*3+2]).float()
   x_rgb = torch.stack([x_rgb1, x_rgb2, x_rgb3], dim=0)

   # x_rgb = x_rgb.unsqueeze(0).float() / 255.0  # BxCxHxW
   x_rgb = transform(x_rgb, params=transform._params)
   print(x_rgb.shape)

   fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
   plt.axis("off")

   img_rgb: np.array = kornia.tensor_to_image(x_rgb)
   # print(img_rgb.shape)
   plt.subplot(1,3,1)
   plt.imshow(img_rgb[0])
   plt.subplot(1,3,2)
   plt.imshow(img_rgb[1])
   plt.subplot(1,3,3)
   plt.imshow(img_rgb[2])
   plt.savefig("test_aug2.png")
   input()