import kornia.augmentation as K
import torch.nn as nn
import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import kornia
import utils

import argparse
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--augmentation_prob', default=[0, 0, 0, 0], nargs='+', type=float, help='get augmentation by probility')
args = parser.parse_args()

transform = nn.Sequential(
   # K.RandomAffine(360),
   K.ColorJitter(0.4)
)

transform = utils.train_diff_transform_prob(*args.augmentation_prob)

for i in range(1000):
   x_rgb: torch.tensor = torchvision.io.read_image('./dog_rgb.png')  # CxHxW / torch.uint8

   x_rgb = x_rgb.unsqueeze(0).float() / 255.0  # BxCxHxW
   x_rgb = transform(x_rgb)
   print(x_rgb.shape)

   fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
   plt.axis("off")

   img_rgb: np.array = kornia.tensor_to_image(x_rgb)
   plt.imshow(img_rgb)
   plt.savefig("test_aug.png")
   input()