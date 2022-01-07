import kornia.augmentation as K
import torch.nn as nn
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import kornia
import pickle
from imageio import imread,imsave

import argparse

parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--file', default='', type=str, help='file to check')
args = parser.parse_args()
# cifar10_1024_4class_grayshift_font_singledigit_mnist
file_path = './data/sampled_cifar10/{}.pkl'.format(args.file)
with open(file_path, "rb") as f:
    data = pickle.load(f)

transform = nn.Sequential(
    # K.RandomResizedCrop([32,32]),
    # K.RandomHorizontalFlip(p=1),
    K.ColorJitter(0.4)
)

print(data.keys())
data_img = data['train_data'].transpose(0,3,1,2)

for i in range(1000):
    x_rgb: torch.tensor = torch.tensor(data_img[i])
    
    x_rgb = x_rgb.unsqueeze(0).float() / 255.0  # BxCxHxW
    img_rgb_org: np.array = kornia.tensor_to_image(x_rgb)
    f = np.fft.fft2(img_rgb_org[:,:,0])
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(1 + fshift))
    
    x_rgb = transform(x_rgb)
    img_rgb: np.array = kornia.tensor_to_image(x_rgb)
    f = np.fft.fft2(img_rgb[:,:,0])
    fshift = np.fft.fftshift(f)
    s2 = np.log(np.abs(1 + fshift))

    s = np.concatenate([s1, s2], axis=1)
    img = np.concatenate([img_rgb_org, img_rgb], axis=1)
    imsave('test_aug.png', img)
    imsave('test.png', s)
    input()