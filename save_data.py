import torch

import numpy as np

from torchvision import utils as vutils

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

data = torch.load('./results/unlearnable_samplewise_52903345_1_20220508015828_0.5_512_2_checkpoint_perturbation_epoch_40.pt').cuda()

# data = np.load('cifar10.npy')

# data = torch.tensor(data).float().permute(0, 3, 1, 2) / 255.0
# print(data.shape)

# mask = torch.ones(data.shape[1:]).cuda()

# print(mask.shape)

# i = 6
# mask[:, :i, :] = 0
# mask[:, 32-i:32, :] = 0
# mask[:, :, 32-i:32] = 0
# mask[:, :, :i] = 0

# mask = mask.unsqueeze(0).repeat(data.shape[0], 1, 1, 1)

# print(mask.shape)

# print(torch.sum(data[:1] != 0))
id_list = [7, 12, 52, 87, 30, 93, 116, 185, 189, 199]
# id_list = [0, 19, 25, 95, 103, 104, 124, 125]

my_data = []
for i in range(data.shape[0]):
    if i not in id_list:
        continue
    
    my_data.append(data[i])
my_data = torch.stack(my_data, dim=0)

my_data = my_data - torch.min(my_data)
my_data = my_data / torch.max(my_data)

for i in range(my_data.shape[0]):

    img = my_data[i:i+1]

    print(torch.min(img))
    print(torch.max(img))

    img_shape = [1, 3, 320, 320]

    resize_img = torch.zeros(size=img_shape)

    for j in range(32):
        for k in range(32):
            resize_img[:,:,j*10:(j+1)*10, k*10:(k+1)*10, ] = img[:,:,j:j+1, k:k+1]

    vutils.save_image(resize_img, '{}.png'.format(id_list[i]))
    # vutils.save_image(resize_img, 'test.png')
    # input(i)

# data_in = data * mask
# data_out = data * (1-mask)

# torch.save(data_in, './results/unlearnable_samplewise_10000_20220518091221_0.5_512_300_checkpoint_perturbation_epoch5_{}_in.pt'.format(i))
# torch.save(data_out, './results/unlearnable_samplewise_10000_20220518091221_0.5_512_300_checkpoint_perturbation_epoch5_{}_out.pt'.format(i))
