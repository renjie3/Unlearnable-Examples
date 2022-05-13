import argparse
import os
import sys

import numpy as np

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model
from utils import train_diff_transform, train_diff_transform2, train_diff_transform_resize48, train_diff_transform_resize64, train_diff_transform_resize28, train_diff_transform_ReCrop_Hflip, train_diff_transform_ReCrop_Hflip_Bri, train_diff_transform_ReCrop_Hflip_Con, train_diff_transform_ReCrop_Hflip_Sat, train_diff_transform_ReCrop_Hflip_Hue, train_transform_no_totensor

import kornia.augmentation as Kaug
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets, metrics

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import time
import math

import matplotlib
from matplotlib.colors import ListedColormap

from torchvision import transforms
import torch.nn.functional as F

import faiss

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print ("check check")

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

def train_simsiam(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], pytorch_aug=False):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    # transform_func = transforms.Compose([
    #     transforms.RandomResizedCrop(32),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     # transforms.ToTensor(),
    #     # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    #     ])

    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      'Tri': utils.train_diff_transform_Tri, 
                      }
    if np.sum(augmentation_prob) == 0:
        if augmentation in transform_func:
            my_transform_func = transform_func[augmentation]
        else:
            raise("Wrong augmentation.")
    else:
        my_transform_func = utils.train_diff_transform_prob(*augmentation_prob)
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = my_transform_func(pos_1), my_transform_func(pos_2)
        # pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # input(pos_1.shape)
    loss = net(pos_1, pos_2)
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()
    # net.update_moving_average()

    # total_num += batch_size
    total_loss = loss.item()
    # print('total_loss', total_loss)
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()

    return total_loss * pos_1.shape[0], pos_1.shape[0]


def train_simsiam_noise_return_loss_tensor(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, dbindex_labels=None, num_clusters=None, single_noise_after_transform=False, no_eval=False, augmentation_prob=None, org_pos1=None, org_pos2=None, clean_weight=0, k_grad=False):
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

    # feature_1, out_1 = net(pos_1)
    # feature_2, out_2 = net(pos_2)
    # print('check0 1')
    
    if not noise_after_transform and not single_noise_after_transform:
        if pytorch_aug:
            # input('check pytorch_aug')
            pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
    else:
        raise('noise_after_transform or single_noise_after_transform to develop')

    time0 = time.time()

    loss = net(pos_1, pos_2, k_grad=k_grad)

    time2 = time.time()

    return loss


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_simsiam(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            # print("data.shape:", data.shape)
            # print("feature.shape:", feature.shape)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        print("test_ssl output DBindex: ")
        print(metrics.davies_bouldin_score(feature_bank.t().contiguous().cpu().numpy(), feature_labels.cpu().numpy()))
        c = np.max(memory_data_loader.dataset.targets) + 1
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100
