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

def train_byol(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], pytorch_aug=False):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    train_transform_no_totensor = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.ToTensor(),
        # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = transform_func(pos_1), transform_func(pos_2)
        # pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # input(pos_1.shape)
    loss = net(pos_1, pos_2)
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()

    return total_loss * pos_1.shape[0], pos_1.shape[0]

def train_simclr_dbindex(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], dbindex_weight=0, pytorch_aug=False, simclr_weight=1, labels=None):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
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

    if dbindex_weight != 0:
        dbindex_loss = get_dbindex_loss(net, pos_1, labels, [10], True, True)
    else:
        dbindex_loss = 0

    if simclr_weight != 0:
        if not noise_after_transform:
            pos_1, pos_2 = my_transform_func(pos_1), my_transform_func(pos_2)
            # pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
        # input(pos_1.shape)

        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
        pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
        pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
        pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
        mask2 = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool()
        # [2*B, 2*B-1]
        neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(2 * pos_1.shape[0], -1)
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)
        
        sim_weight, sim_indices = neg_sim_matrix2.topk(k=10, dim=-1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        
        simclr_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    else:
        simclr_loss = 0
    
    loss = simclr_loss * simclr_weight + dbindex_loss * dbindex_weight

    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()

    if simclr_weight != 0:
        return total_loss * pos_1.shape[0], pos_1.shape[0], torch.log(pos_sim.mean()).item() / 2, torch.log(sim_weight.mean()).item() / 2
    else:
        return total_loss * pos_1.shape[0], pos_1.shape[0], 0, 0

def train_simclr_theory(net, pos_1, pos_2, train_optimizer, batch_size, temperature, random_drop_feature_num, gaussian_aug_std=0.05, thoery_schedule_dim=90, theory_aug_by_order=False):
    net.train()
    total_loss, total_num = 0.0, 0
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True)[:,0,:,0], pos_2.cuda(non_blocking=True)[:,0,:,0]
    
    drop_mask1 = []
    drop_mask2 = []
    level = len(random_drop_feature_num)
    if thoery_schedule_dim == 90:
        level_dim = [0, 10, 30, 50, 70, 90]
    elif thoery_schedule_dim == 150:
        level_dim = [0, 10, 30, 60, 100, 150]
    elif thoery_schedule_dim == 20:
        level_dim = [0, 10, 20]
    elif thoery_schedule_dim == 30:
        level_dim = [0, 10, 20, 30]
    elif thoery_schedule_dim == 10:
        level_dim = [0, 10]
    elif thoery_schedule_dim == 50:
        level_dim = [0, 10, 20, 30, 40, 50]
    else:
        raise("Wrong thoery_schedule_dim!")
    gaussian_schedule = [1, 0, 0, 0, 0]
    # s_feature_dim = 10
    if theory_aug_by_order:
        # input('check here')
        for i in range(0, level):
            ids = np.arange(level_dim[i], level_dim[i+1])
            drop_feature = ids[:random_drop_feature_num[i]]
            drop_mask1.append(drop_feature)
            drop_feature = ids[:random_drop_feature_num[i]]
            drop_mask2.append(drop_feature)
    else:
        for i in range(0, level):
            ids = np.arange(level_dim[i], level_dim[i+1])
            random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
            drop_mask1.append(random_frop_feature)
            random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
            drop_mask2.append(random_frop_feature)
    # input(drop_mask1)
    drop_mask1 = np.concatenate(drop_mask1, axis=0)
    drop_mask2 = np.concatenate(drop_mask2, axis=0)
    aug1 = np.ones(pos_1.shape[1])
    for drop_feat in drop_mask1:
        aug1[drop_feat] = 0.0
    aug2 = np.ones(pos_2.shape[1])
    for drop_feat in drop_mask2:
        aug2[drop_feat] = 0.0
    aug1 = torch.tensor(aug1).cuda(non_blocking=True)
    aug2 = torch.tensor(aug2).cuda(non_blocking=True)
    # input(drop_mask1)
    pos_1 = (pos_1 * aug1).float()
    pos_2 = (pos_2 * aug2).float()
    if gaussian_aug_std != 0:
        gaussian_aug1 = []
        gaussian_aug2 = []
        for i in range(0, level):
            mean = torch.tensor([0 for _ in range(level_dim[i+1] - level_dim[i])]).float()
            std = torch.tensor([gaussian_aug_std * gaussian_schedule[i] for _ in range(level_dim[i+1] - level_dim[i])]).float()
            aug_batch1 = []
            aug_batch2 = []
            for _ in range(pos_1.shape[0]):
                aug_batch1.append(torch.normal(mean, std).cuda(non_blocking=True))
                aug_batch2.append(torch.normal(mean, std).cuda(non_blocking=True))
            gaussian_aug1.append(torch.stack(aug_batch1, dim=0))
            gaussian_aug2.append(torch.stack(aug_batch2, dim=0))
            # gaussian_aug1.append(torch.normal(mean, std).cuda(non_blocking=True))
            # gaussian_aug2.append(torch.normal(mean, std).cuda(non_blocking=True))
            # print(torch.normal(mean, std).cuda(non_blocking=True).shape)
            # print(gaussian_aug1)
            # input(gaussian_aug2)
        gaussian_aug1 = torch.cat(gaussian_aug1, dim=1)
        gaussian_aug2 = torch.cat(gaussian_aug2, dim=1)
        pos_1 += gaussian_aug1
        pos_2 += gaussian_aug2

        # for i in range(0, level):
        #     mean = torch.tensor([0 for _ in range(level_dim[i+1] - level_dim[i])]).float()
        #     std = torch.tensor([gaussian_aug_std * gaussian_schedule[i] for _ in range(level_dim[i+1] - level_dim[i])]).float()
        #     aug_batch1 = []
        #     aug_batch2 = []
        #     gaussian_aug1.append(torch.normal(mean, std).cuda(non_blocking=True))
        #     gaussian_aug2.append(torch.normal(mean, std).cuda(non_blocking=True))
        #     # print(torch.normal(mean, std).cuda(non_blocking=True).shape)
        # gaussian_aug1 = torch.cat(gaussian_aug1, dim=0)
        # gaussian_aug2 = torch.cat(gaussian_aug2, dim=0)
        # pos_1 += gaussian_aug1
        # pos_2 += gaussian_aug2

        # test_aug = torch.tensor([0 for _ in range(10)] + [1 for _ in range(20)]).cuda(non_blocking=True)
        # pos_1 *= test_aug
        # pos_2 *= test_aug

    out_1 = net(pos_1)
    out_2 = net(pos_2)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
    pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
    pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
    mask2 = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool()
    # [2*B, 2*B-1]
    neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(2 * pos_1.shape[0], -1)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)
    
    sim_weight, sim_indices = neg_sim_matrix2.topk(k=10, dim=-1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()

    return total_loss * pos_1.shape[0], pos_1.shape[0], torch.log(pos_sim.mean()).item() / 2, torch.log(sim_weight.mean()).item() / 2

def train_simclr_newneg(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0]):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      'Bri': utils.train_diff_transform_Bri,
                      }
    if np.sum(augmentation_prob) == 0:
        if augmentation in transform_func:
            my_transform_func = transform_func[augmentation]
        else:
            raise("Wrong augmentation.")
    else:
        my_transform_func = utils.train_diff_transform_prob(*augmentation_prob)
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    x = pos_1
    if not noise_after_transform:
        if mix in ['concat_samplewise_train_mnist_18_128', 'concat_samplewise_all_mnist_18_128']:
            pos_1, pos_2 = train_diff_transform_resize48(pos_1), train_diff_transform_resize48(pos_2)
        elif mix in ['concat4_samplewise_train_mnist_18_128', 'concat4_samplewise_all_mnist_18_128']:
            pos_1, pos_2 = train_diff_transform_resize64(pos_1), train_diff_transform_resize64(pos_2)
        elif mix in ['mnist']:
            pos_1, pos_2 = train_diff_transform_resize28(pos_1), train_diff_transform_resize28(pos_2)
        else:
            pos_1, pos_2 = my_transform_func(pos_1), my_transform_func(pos_2)
        # pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # print(pos_1.shape)
    feature_x, out_x = net(x)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, B]
    sim_matrix = torch.exp(torch.mm(out, out_x.t().contiguous()) / temperature)
    # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # # -----pos neg similarity
    # pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
    # pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
    # pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
    # mask2 = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool()
    # # [2*B, 2*B-1]
    # neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(2 * pos_1.shape[0], -1)
    # # -----
    # sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)
    
    sim_weight, sim_indices = sim_matrix.topk(k=10, dim=-1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    
    return total_loss * pos_1.shape[0], pos_1.shape[0], torch.log(pos_sim.mean()).item() / 2, torch.log(sim_weight.mean()).item() / 2

def train_simclr_2digit(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], batchsize_2digit=256):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      'Bri': utils.train_diff_transform_Bri,
                      }
    if np.sum(augmentation_prob) == 0:
        if augmentation in transform_func:
            my_transform_func = transform_func[augmentation]
        else:
            raise("Wrong augmentation.")
    else:
        my_transform_func = utils.train_diff_transform_prob(*augmentation_prob)
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    
    aug1_part1 = pos_1[:batchsize_2digit]
    aug1_part2 = pos_1[batchsize_2digit:2*batchsize_2digit]
    aug2_part1 = pos_1[2*batchsize_2digit:3*batchsize_2digit]
    aug2_part2 = pos_1[3*batchsize_2digit:4*batchsize_2digit]
    
    aug1 = torch.cat([aug1_part1, aug1_part2], dim=0)
    aug2 = torch.cat([aug2_part1, aug2_part2], dim=0)
    
    feature_1, out_1 = net(aug1)
    feature_2, out_2 = net(aug2)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out_1, out_2.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batchsize_2digit, device=sim_matrix.device)).bool()
    # # -----pos neg similarity
    # pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
    # pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
    # pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
    # mask2 = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool()
    # # [2*B, 2*B-1]
    # neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(2 * pos_1.shape[0], -1)
    # # -----
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batchsize_2digit, -1)
    
    sim_weight, sim_indices = sim_matrix.topk(k=10, dim=-1)

    # compute loss
    # [2*B]
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    
    return total_loss * pos_1.shape[0], pos_1.shape[0], torch.log(pos_sim.mean()).item() / 2, torch.log(sim_weight.mean()).item() / 2

def train_simclr_softmax(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0]):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      }
    if np.sum(augmentation_prob) == 0:
        if augmentation in transform_func:
            my_transform_func = transform_func[augmentation]
        else:
            raise("Wrong augmentation.")
    else:
        my_transform_func = utils.train_diff_transform_prob(*augmentation_prob)
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if not noise_after_transform:
        if mix in ['concat_samplewise_train_mnist_18_128', 'concat_samplewise_all_mnist_18_128']:
            pos_1, pos_2 = train_diff_transform_resize48(pos_1), train_diff_transform_resize48(pos_2)
        elif mix in ['concat4_samplewise_train_mnist_18_128', 'concat4_samplewise_all_mnist_18_128']:
            pos_1, pos_2 = train_diff_transform_resize64(pos_1), train_diff_transform_resize64(pos_2)
        elif mix in ['mnist']:
            pos_1, pos_2 = train_diff_transform_resize28(pos_1), train_diff_transform_resize28(pos_2)
        else:
            pos_1, pos_2 = my_transform_func(pos_1), my_transform_func(pos_2)
        # pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # print(pos_1.shape)
    feature_1, logits_1, out_1 = net(pos_1)
    feature_2, logits_2, out_2 = net(pos_2)
    
    # [2*B, D]
    logits = torch.cat([logits_1, logits_2], dim=0)
    criterion = torch.nn.CrossEntropayLoss().to(feature_1.device)
    targets = torch.cat([torch.arange(pos_1.shape[0]), torch.arange(pos_1.shape[0])], dim = 0).to(feature_1.device)
    
    loss = criterion(logits, targets)
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()

    return total_loss * pos_1.shape[0], pos_1.shape[0]

def train_simclr_target_task(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, target_task="pos/neg"):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if not noise_after_transform:
        pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    # [[0,I], [I,0]] mask to remove positive pairs in denominator
    pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
    pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
    pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool() # here it didn't remove the similarity between positive samples. We changed it into remove that.
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # print(out_1.shape) torch.Size([512, 128])
    # print(out_2.shape) torch.Size([512, 128])

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    if target_task == "pos/neg":
        loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1)))).mean()
    elif target_task == "pos":
        loss = (- torch.log(pos_sim)).mean()
    elif target_task == "neg":
        loss = (- torch.log(1 / sim_matrix.sum(dim=-1))).mean()
    numerator = pos_sim.mean().item()
    denominator = sim_matrix.sum(dim=-1).mean().item()

    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    # print(pos_sim.shape)
    # print(sim_matrix.sum(dim=-1).shape)
    # input()
    
    return total_loss * pos_1.shape[0], pos_1.shape[0], numerator, denominator


def get_linear_noise_dbindex_loss(x, labels, use_mean_dbindex=True, use_normalized=True, noise_centroids=None, modify_dbindex=''):

    sample = x.reshape(x.shape[0], -1)
    cluster_label = labels

    class_center = []
    class_center_wholeset = []
    intra_class_dis = []
    c = torch.max(cluster_label) + 1
    # print(c)
    # print("time2: {}".format(time2 - time1))
    for i in range(c):
        # print(i)
        idx_i = torch.where(cluster_label == i)[0]
        if idx_i.shape[0] == 0:
            continue
        class_i = sample[idx_i, :]

        if noise_centroids == None:
            if use_normalized:
                class_i_center = nn.functional.normalize(class_i.mean(dim=0), p=2, dim=0)
            else:
                class_i_center = class_i.mean(dim=0)
        else:
            class_i_center = noise_centroids[i].cuda()

        class_center.append(class_i.mean(dim=0))

        point_dis_to_center = torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))

        intra_class_dis.append(torch.mean(point_dis_to_center))

    # print("time3: {}".format(time3 - time2))
    if len(class_center) <= 1:
        return 0
    class_center = torch.stack(class_center, dim=0)

    c = len(intra_class_dis)
    
    class_dis = torch.cdist(class_center, class_center, p=2) # TODO: this can be done for only one time in the whole set

    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

    if use_mean_dbindex:
        cluster_DB_loss = ((intra_class_dis_pair_sum + 0.00001) / (class_dis + 0.00001)).mean()
    else:
        cluster_DB_loss = torch.max((intra_class_dis_pair_sum + 0.00001) / (class_dis + 0.00001), dim=1)[0].mean()
    
    if modify_dbindex == '':
        loss = cluster_DB_loss
    elif modify_dbindex == '_e_B':
        loss = -torch.exp(- cluster_DB_loss)
    elif modify_dbindex == 'e_B':
        loss = torch.exp(cluster_DB_loss)
    elif modify_dbindex == 'log_B':
        loss = torch.log(cluster_DB_loss)
    elif modify_dbindex == 'log_B1':
        loss = torch.log(cluster_DB_loss + 1)

    print('get_linear_noise_dbindex_loss:', cluster_DB_loss.item())

    return loss

def get_dbindex_loss(net, x, labels, num_clusters, use_out_dbindex, use_mean_dbindex, dbindex_label_index, x2, use_aug):

    time0 = time.time()

    if use_aug:
        pos_1 = train_diff_transform(x)
        pos_2 = train_diff_transform(x2)
        x = torch.cat([pos_1, pos_2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

    feature, out = net(x)
    if use_out_dbindex:
        sample = out.double()
    else:
        sample = feature.double()

    time1 = time.time()
    # print("time1: {}".format(time1 - time0))

    loss = 0
    n_clueter_num = len(num_clusters)
    for num_cluster_idx in range(len(num_clusters)):
        cluster_label = labels[:, dbindex_label_index]
        # cluster_label = cluster_label.repeat((repeat_num, ))

        class_center = []
        class_center_wholeset = []
        intra_class_dis = []
        c = torch.max(cluster_label) + 1
        time2 = time.time()
        # print(c)
        # print("time2: {}".format(time2 - time1))
        for i in range(c):
            # print(i)
            idx_i = torch.where(cluster_label == i)[0]
            if idx_i.shape[0] == 0:
                continue
            class_i = sample[idx_i, :]

            class_i_center = nn.functional.normalize(class_i.mean(dim=0), p=2, dim=0)

            class_center.append(class_i_center)

            point_dis_to_center = torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))

            intra_class_dis.append(torch.mean(point_dis_to_center))
        time3 = time.time()
        # print("time3: {}".format(time3 - time2))
        if len(class_center) <= 1:
            continue
        class_center = torch.stack(class_center, dim=0)
        # input('no')

        time4 = time.time()
        # print("time4: {}".format(time4 - time3))

        c = len(intra_class_dis)
        
        class_dis = torch.cdist(class_center, class_center, p=2) # TODO: this can be done for only one time in the whole set

        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        time5 = time.time()
        # print("time5: {}".format(time5 - time4))

        # cod_x, cod_y = torch.where(class_dis == 0)
        # for x, y in zip(cod_x, cod_y):
        #     print(x,y)
        #     print(class_center[x])
        #     print(class_center[y])
        #     print(class_center[x] - class_center[y])
        #     print((class_center[x] - class_center[y])**2)
        #     print(torch.sum((class_center[x] - class_center[y])**2))
        #     print(x,y)
        #     input()

        if use_mean_dbindex:
            cluster_DB_loss = (intra_class_dis_pair_sum / (class_dis + 0.00001)).mean()
        else:
            cluster_DB_loss = torch.max(intra_class_dis_pair_sum / (class_dis + 0.00001), dim=1)[0].mean()

        time6 = time.time()
        # print("time6: {}".format(time6 - time5))

        loss += cluster_DB_loss

    # input('time done')

    return loss

def run_kmeans(x, num_clusters, device, temperature):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(num_clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        # centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()
                density[i] = d     
                
        # #if cluster only has one point, use the max to estimate its concentration        
        # dmax = density.max()
        # for i,dist in enumerate(Dcluster):
        #     if len(dist)<=1:
        #         density[i] = dmax

        # density = density.clip(0, np.percentile(density,90)) #clamp extreme values for stability
        # density = temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        # centroids = torch.Tensor(centroids).cuda()
        # centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        # results['centroids'].append(centroids)
        # results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def find_cluster(net, memory_data_loader, random_noise, n_components, label_index=0, use_feature_find_cluster=False):

    time0 = time.time()

    print('Finding clusters')

    feature_bank = []
    net.eval()

    # plot_num = 2048
    # n_components = 11

    with torch.no_grad():

        for pos_samples_1, pos_samples_2, labels in memory_data_loader:
            pos_samples_1, pos_samples_2, labels = pos_samples_1.cuda(), pos_samples_2.cuda(), labels.cuda()

            train_pos_1 = []
            train_pos_2 = []
            for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                sample_noise = random_noise[label[label_index].item()]
                if type(sample_noise) is np.ndarray:
                    mask = sample_noise
                else:
                    mask = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).cuda()
                train_pos_1.append(pos_samples_1[i]+sample_noise)
                train_pos_2.append(pos_samples_2[i]+sample_noise)

            train_pos_1 = torch.stack(train_pos_1, dim=0)
            train_pos_2 = torch.stack(train_pos_2, dim=0)

            feature, out = net(train_pos_1)
            if use_feature_find_cluster:
                feature_bank.append(feature)
            else:
                feature_bank.append(out)

        feature_bank = torch.cat(feature_bank, dim=0) #[:plot_num]
        # sim_matrix = torch.matmul(feature_bank, feature_bank.t()) - torch.eye(feature_bank.shape[0], device=feature_bank.device)

        # sim_matrix = torch.cdist(feature_bank, feature_bank)
        # # print(sim_matrix)
        # # print(sim_matrix.shape)

        # # adj = (sim_matrix > 0.995).detach().cpu().numpy().astype(int) #.long()
        # adj = (sim_matrix < 0.1).detach().cpu().numpy().astype(int) #.long()

        # print(adj)

        kmeans_result = run_kmeans(feature_bank.detach().cpu().numpy(), [n_components], 0, 0.5)

    # adj = csr_matrix(adj)
    # n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    # print(n_components, labels)
    # print(type(labels))
    # print(len(labels))
    # for i in range(n_components):
    #     idx = np.where(labels == i)[0]
    #     print(idx)
    #     print(len(idx))
    #     input()

    labels = kmeans_result['im2cluster'][0].detach().cpu().numpy()

    # cm = plt.cm.get_cmap('gist_rainbow', n_components)
    # # z = np.arange(n_components)
    # # my_cmap = cm(z)
    # # my_cmap = ListedColormap(my_cmap)
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    # feature_tsne_input = feature_bank.detach().cpu().numpy()[:plot_num]
    # labels_tsne_color = labels[:plot_num]

    # feature_tsne_output = tsne.fit_transform(feature_tsne_input)
    # coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
    # coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

    # for i in range(n_components):
    #     idx = np.where(labels == i)[0]

    #     fig = plt.figure(figsize=(8, 8))
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.title("Find {} clusters.".format(n_components))
    #     plt.xlim((coord_min, coord_max))
    #     plt.ylim((coord_min, coord_max))
    #     plt.scatter(feature_tsne_output[idx, 0], feature_tsne_output[idx, 1], s=10, c=labels_tsne_color[idx], cmap=my_cmap)
    #     plt.savefig('./visualization/find_cluster.png')
    #     plt.close()
    #     input()

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.title("Find {} clusters.".format(n_components))
    # plt.xlim((coord_min, coord_max))
    # plt.ylim((coord_min, coord_max))
    # plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=cm)
    # plt.savefig('./visualization/find_cluster.png')
    # plt.close()

    # time1 = time.time()
    # print("finding clusters: {}".format(time1 - time0))

    # input('check time')

    return labels

def train_simclr_noise_return_loss_tensor(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, dbindex_labels=None, num_clusters=None, single_noise_after_transform=False, no_eval=False, augmentation_prob=None, org_pos1=None, org_pos2=None, clean_weight=0):
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if clean_weight != 0 and org_pos1 != None and org_pos2 != None:
        org_pos1, org_pos2 = org_pos1.cuda(non_blocking=True), org_pos2.cuda(non_blocking=True)

    # feature_1, out_1 = net(pos_1)
    # feature_2, out_2 = net(pos_2)
    # print('check0 1')
    
    if not noise_after_transform and not single_noise_after_transform:

        if split_transform:
            # print('check split_transform')
            pos_1_list = torch.split(pos_1, batch_size // 16, dim=0)
            pos_2_list = torch.split(pos_2, batch_size // 16, dim=0)
            new_pos_1_list, new_pos_2_list = [], []
            for sub_pos_1 in pos_1_list:
                # print(sub_pos_1.shape)
                if pytorch_aug:
                    new_pos_1_list.append(train_transform_no_totensor(sub_pos_1))
                else:
                    new_pos_1_list.append(train_diff_transform(sub_pos_1))
            for sub_pos_2 in pos_2_list:
                # print(sub_pos_2.shape)
                if pytorch_aug:
                    new_pos_2_list.append(train_transform_no_totensor(sub_pos_2))
                else:
                    new_pos_2_list.append(train_diff_transform(sub_pos_2))
            pos_1 = torch.cat(new_pos_1_list, dim=0)
            pos_2 = torch.cat(new_pos_2_list, dim=0)
        else:
            if pytorch_aug:
                # input('check pytorch_aug')
                pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
            else:
                # print('chech right aug')
                if augmentation_prob == None:
                    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
                    if clean_weight != 0 and org_pos1 != None and org_pos2 != None:
                        org_pos1, org_pos2 = train_diff_transform(org_pos1), train_diff_transform(org_pos2)
                elif np.sum(augmentation_prob) == 0:
                    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
                    if clean_weight != 0 and org_pos1 != None and org_pos2 != None:
                        org_pos1, org_pos2 = train_diff_transform(org_pos1), train_diff_transform(org_pos2)
                else:
                    my_diff_transform = utils.train_diff_transform_prob(*augmentation_prob)
                    pos_1, pos_2 = my_diff_transform(pos_1), my_diff_transform(pos_2)
                    if clean_weight != 0 and org_pos1 != None and org_pos2 != None:
                        org_pos1, org_pos2 = my_diff_transform(org_pos1), my_diff_transform(org_pos2)

    if single_noise_after_transform:
        pos_2 = train_diff_transform(pos_2)

    if not no_eval:
        net.eval()

    time0 = time.time()

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    time1 = time.time()

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    if clean_weight != 0:
        org_feature_1, org_out_1 = net(org_pos1)
        org_feature_2, org_out_2 = net(org_pos2)
        org_out = torch.cat([org_out_1, org_out_2], dim=0)
        poison_clean_pos_sim = torch.exp(torch.sum(out * org_out, dim=-1) / temperature).mean()
        loss += clean_weight * poison_clean_pos_sim
        # print(poison_clean_pos_sim.item())
        # input('check')
    
    # train_optimizer.zero_grad()
    # perturb.retain_grad()
    # loss.backward()

    time2 = time.time()

    return loss

def train_simclr_noise_return_loss_tensor_no_eval(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, dbindex_labels=None, num_clusters=None):
    # net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

    # feature_1, out_1 = net(pos_1)
    # feature_2, out_2 = net(pos_2)
    # print('check0 1')
    
    if not noise_after_transform:
        if split_transform:
            pass
            pos_1_list = torch.split(pos_1, batch_size // 16, dim=0)
            pos_2_list = torch.split(pos_2, batch_size // 16, dim=0)
            new_pos_1_list, new_pos_2_list = [], []
            for sub_pos_1 in pos_1_list:
                # print(sub_pos_1.shape)
                if pytorch_aug:
                    new_pos_1_list.append(train_transform_no_totensor(sub_pos_1))
                else:
                    new_pos_1_list.append(train_diff_transform(sub_pos_1))
            for sub_pos_2 in pos_2_list:
                # print(sub_pos_2.shape)
                if pytorch_aug:
                    new_pos_2_list.append(train_transform_no_totensor(sub_pos_2))
                else:
                    new_pos_2_list.append(train_diff_transform(sub_pos_2))
            pos_1 = torch.cat(new_pos_1_list, dim=0)
            pos_2 = torch.cat(new_pos_2_list, dim=0)
        else:
            if pytorch_aug:
                # input('check pytorch_aug')
                pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
            else:
                pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)

    # feature_1, out_1 = net(pos_1)
    # print('check0 2')

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    # train_optimizer.zero_grad()
    # perturb.retain_grad()
    # loss.backward()

    return loss

def train_simclr_noise_return_loss_tensor_no_eval_pos_only(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False, split_transform=False, pytorch_aug=False, dbindex_weight=0, dbindex_labels=None, num_clusters=None):
    # net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

    # feature_1, out_1 = net(pos_1)
    # feature_2, out_2 = net(pos_2)
    # print('check0 1')
    
    if not noise_after_transform:
        if split_transform:
            pass
            pos_1_list = torch.split(pos_1, batch_size // 16, dim=0)
            pos_2_list = torch.split(pos_2, batch_size // 16, dim=0)
            new_pos_1_list, new_pos_2_list = [], []
            for sub_pos_1 in pos_1_list:
                # print(sub_pos_1.shape)
                if pytorch_aug:
                    new_pos_1_list.append(train_transform_no_totensor(sub_pos_1))
                else:
                    new_pos_1_list.append(train_diff_transform(sub_pos_1))
            for sub_pos_2 in pos_2_list:
                # print(sub_pos_2.shape)
                if pytorch_aug:
                    new_pos_2_list.append(train_transform_no_totensor(sub_pos_2))
                else:
                    new_pos_2_list.append(train_diff_transform(sub_pos_2))
            pos_1 = torch.cat(new_pos_1_list, dim=0)
            pos_2 = torch.cat(new_pos_2_list, dim=0)
        else:
            if pytorch_aug:
                # input('check pytorch_aug')
                pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
            else:
                pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)

    # feature_1, out_1 = net(pos_1)
    # print('check0 2')

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / 2)).mean()
    
    # train_optimizer.zero_grad()
    # perturb.retain_grad()
    # loss.backward()

    return loss

def train_simclr_noise_return_loss_tensor_full_gpu(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug, noise_after_transform, gpu_times, cross_eot):
    net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    # if flag_strong_aug:
    #     pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # else:
    #     pos_1, pos_2 = train_diff_transform2(pos_1), train_diff_transform2(pos_2)
    if not noise_after_transform:
        pos_1_list = []
        pos_2_list = []
        for _ in range(gpu_times):
            pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
            pos_1_list.append(pos_1)
            pos_2_list.append(pos_2)
        pos_1 = torch.cat(pos_1_list, dim=0)
        pos_2 = torch.cat(pos_2_list, dim=0)
    
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    out_1_list = torch.split(out_1, batch_size, dim=0)
    out_2_list = torch.split(out_2, batch_size, dim=0)

    total_loss = 0
    count_loss = 0

    if cross_eot:

        for out_1 in out_1_list:
            for out_2 in out_2_list:
                # [2*B, D]
                out = torch.cat([out_1, out_2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
                mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
                # [2*B, 2*B-1]
                sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)

                # compute loss
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                total_loss += loss
                count_loss += 1

    else:

        for out_1, out_2 in zip(out_1_list, out_2_list):

            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            total_loss += loss
            count_loss += 1

    total_loss /= count_loss

    return total_loss

def train_simclr_noise_return_loss_tensor_model_free(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False):
    net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if not noise_after_transform:
        os_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    # train_optimizer.zero_grad()
    # perturb.retain_grad()
    # loss.backward()

    return loss

def train_simclr_noise(net, pos_samples_1, pos_samples_2, perturb, train_optimizer, batch_size, temperature):
    # train a batch
    # print(pos_1.shape)
    # print(pos_2.shape)
    # print("batch_size", batch_size)
    # print("this is train_simclr_noise")
    pos_1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
    pos_2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    
    net.eval()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    # train_optimizer.zero_grad()
    perturb.retain_grad()
    loss.backward()
    # train_optimizer.step() # step is used to update Variable. But we do not update like SGD. We use the sign

    # total_num += batch_size
    total_loss = loss.item()
    # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / pos_1.shape[0]

def train_simclr_noise_return_loss_tensor_target_task(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, target_task="pos/neg"):
    # train a batch
    # print(pos_1.shape) torch.Size([512, 3, 32, 32])
    # print(pos_2.shape) torch.Size([512, 3, 32, 32])
    # print("batch_size", batch_size)
    # print("this is train_simclr_noise")
    # pos_1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
    # pos_2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
    net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if flag_strong_aug:
        pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    else:
        pos_1, pos_2 = train_diff_transform2(pos_1), train_diff_transform2(pos_2)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)

    if "neg" in target_task:
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        # [[0,I], [I,0]] mask to remove positive pairs in denominator
        pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device), torch.eye(pos_1.shape[0], device=sim_matrix.device)], dim=0)
        pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=sim_matrix.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=sim_matrix.device)], dim=0)
        pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device) - pos_den_mask).bool() # here it didn't remove the similarity between positive samples. We changed it into remove that.
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # print(out_1.shape) torch.Size([512, 128])
    # print(out_2.shape) torch.Size([512, 128])

    # compute loss
    if "pos" in target_task:
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    if target_task == "pos/neg":
        loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1)))).mean()
    elif target_task == "pos":
        loss = (- torch.log(pos_sim)).mean()
    elif target_task == "neg":
        loss = (- torch.log(1 / sim_matrix.sum(dim=-1))).mean()
    # train_optimizer.zero_grad()
    # perturb.retain_grad()
    # loss.backward()

    return loss

def train_simclr_noise_return_loss_tensor_eot(net, pos_1, pos_2, train_optimizer, batch_size, temperature, eot_size, flag_strong_aug = True):
    # train a batch
    # print(pos_1.shape)
    # print(pos_2.shape)
    # print("batch_size", batch_size)
    # print("this is train_simclr_noise")
    # pos_1 = torch.clamp(pos_samples_1.data + perturb, 0, 1)
    # pos_2 = torch.clamp(pos_samples_2.data + perturb, 0, 1)
    net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = torch.cat(pos_1, dim=0), torch.cat(pos_2, dim=0)
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    if flag_strong_aug:
        pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    else:
        pos_1, pos_2 = train_diff_transform2(pos_1), train_diff_transform2(pos_2)
    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    eot_out_1 = torch.chunk(out_1, eot_size, dim=0)
    eot_out_2 = torch.chunk(out_2, eot_size, dim=0)

    loss = 0
    for i in range(eot_size):
        # [2*B, D]
        out = torch.cat([eot_out_1[i], eot_out_2[i]], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * eot_out_1[i].shape[0], device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * eot_out_1[i].shape[0], -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(eot_out_1[i] * eot_out_2[i], dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss += (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    return loss / eot_size

# def train_simclr_noise_pos1_pertub(net, pos_1, pos_2, train_optimizer, batch_size, temperature):
#     # train a batch
#     # print(pos_1.shape)
#     # print(pos_2.shape)
#     # print("batch_size", batch_size)
#     # print("this is train_simclr_noise")
#     # pos_1 = torch.clamp(pos_samples_1.data, 0, 1)
#     # pos_2 = torch.clamp(pos_samples_2.data, 0, 1)
#     net.eval()
#     total_loss, total_num = 0.0, 0
#     # for pos_1, pos_2, target in train_bar:
#     pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
#     pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
#     feature_1, out_1 = net(pos_1)
#     feature_2, out_2 = net(pos_2)
#     # [2*B, D]
#     out = torch.cat([out_1, out_2], dim=0)
#     # [2*B, 2*B]
#     sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#     mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
#     # [2*B, 2*B-1]
#     sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

#     # compute loss
#     pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#     # [2*B]
#     pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
#     loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
#     train_optimizer.zero_grad()
#     pos_1.retain_grad()
#     loss.backward()
#     # train_optimizer.step() # step is used to update Variable. But we do not update like SGD. We use the sign of gradients

#     # total_num += batch_size
#     total_loss = loss.item()
#     # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

#     return total_loss / pos_1.shape[0]

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_ssl_theory(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs, c=10):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            data = data.cuda(non_blocking=True)[:,0,:,0]
            out = net(data)
            feature_bank.append(out)
            # print("data.shape:", data.shape)
            # print("feature.shape:", feature.shape)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True)[:,0,:,0], target.cuda(non_blocking=True)
            feature = net(data)

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

    return total_top1 / total_num * 100, total_top5 / total_num * 100, feature_bank.t().contiguous(), feature_labels

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_ssl(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
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
            feature, out = net(data)

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

def test_intra_inter_sim(net, memory_data_loader, test_data_loader, k, temperature, epochs, distance=False, c=10, theory_model=False):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(test_data_loader, desc='Feature extracting'):
            if theory_model:
                feature = net(data.cuda(non_blocking=True)[:,0,:,0])
            else:
                feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            # print("data.shape:", data.shape)
            # print("feature.shape:", feature.shape)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        # [N]
        feature_labels = torch.tensor(test_data_loader.dataset.targets, device=feature_bank.device)
        c = max(feature_labels) + 1
        # loop test data to predict the label by weighted knn search
        intra_sim_list = []
        class_center = []
        for i in range(c):
            i_sub = torch.where(feature_labels == i)
            i_sub = i_sub[0]
            # [i_B, D]
            i_data = feature_bank[i_sub]
            # [D]
            i_center = i_data.mean(dim=0)#.unsqueeze(1)
            # [i_B]
            if distance:
                intra_sim = torch.norm(i_data - i_center, p = 2, dim=1).mean()
            else:
                intra_sim = torch.mm(i_data, i_center.unsqueeze(1)).mean()
            intra_sim_list.append(intra_sim.item())
            class_center.append(i_center)
        class_center = torch.stack(class_center, dim=0)
        intra_sim = np.mean(intra_sim_list)
        if distance:
            inter_dis_list = []
            for i in range(c):
                for j in range(i+1,c):
                    # input(torch.norm(class_center[i] - class_center[j], p = 2, dim=0))
                    inter_dis_list.append(torch.norm(class_center[i] - class_center[j], p = 2, dim=0).item())
            inter_sim = np.mean(inter_dis_list)
        else:
            inter_sim = torch.mm(class_center, class_center.t().contiguous())
            mask = (torch.ones_like(inter_sim) - torch.eye(c, device=inter_sim.device)).bool()
            inter_sim = inter_sim.masked_select(mask).view(c, -1).mean().item()

    return intra_sim, inter_sim

def test_cluster(net, memory_data_loader, test_data_loader, dim_range, random_drop_feature_num, gaussian_aug_std, theory_aug_by_order=False, thoery_schedule_dim=30, flag_output_cluster=False, theory_model=False, instance_level=False):
    # dim_range = np.arange(a, b)
    feature_labels = test_data_loader.dataset.targets
    if instance_level:
        feature_labels = np.arange(0, feature_labels.shape[0])
    return_values = {}
    if flag_output_cluster:
        net.eval()
        feature_bank = []
        label_bank = []
        # c = 10
        with torch.no_grad():
            # generate feature bank
            for data, _, target in tqdm(test_data_loader, desc='Feature extracting'):
                data = data.cuda(non_blocking=True)[:,0,:,0]
                if thoery_schedule_dim == 90:
                    level_dim = [0, 10, 30, 50, 70, 90]
                elif thoery_schedule_dim == 150:
                    level_dim = [0, 10, 30, 60, 100, 150]
                elif thoery_schedule_dim == 20:
                    level_dim = [0, 10, 20]
                elif thoery_schedule_dim == 30:
                    level_dim = [0, 10, 20, 30]
                elif thoery_schedule_dim == 10:
                    level_dim = [0, 10]
                elif thoery_schedule_dim == 50:
                    level_dim = [0, 10, 20, 30, 40, 50]
                else:
                    raise("Wrong thoery_schedule_dim!")
                level = len(level_dim) - 1
                gaussian_schedule = [1, 0, 0, 0, 0]
                drop_mask = []
                if theory_aug_by_order:
                    # input('check here')
                    for i in range(0, level):
                        ids = np.arange(level_dim[i], level_dim[i+1])
                        drop_feature = ids[:random_drop_feature_num[i]]
                        drop_mask.append(drop_feature)
                else:
                    for i in range(0, level):
                        ids = np.arange(level_dim[i], level_dim[i+1])
                        random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
                        drop_mask.append(random_frop_feature)
                # input(drop_mask1)
                drop_mask = np.concatenate(drop_mask, axis=0)

                aug = np.ones(data.shape[1])
                for drop_feat in drop_mask:
                    aug[drop_feat] = 0.0
                # print(np.sum(aug))
                aug_bank = data * torch.tensor(aug).cuda(non_blocking=True)
                if gaussian_aug_std != 0:
                    gaussian_aug = []
                    for i in range(0, level):
                        mean = np.array([0 for _ in range(level_dim[i+1] - level_dim[i])])
                        std = np.diag([gaussian_aug_std * gaussian_schedule[i] for _ in range(level_dim[i+1] - level_dim[i])])
                        aug_noise = np.random.multivariate_normal(mean=mean, cov=std, size=data.shape[0])
                        gaussian_aug.append(aug_noise)
                    gaussian_aug = np.concatenate(gaussian_aug, axis=1)
                    gaussian_aug = torch.tensor(gaussian_aug).cuda(non_blocking=True)
                    aug_bank += gaussian_aug
                    # gaussian_aug.repeat()

                label_bank.append(target.cuda(non_blocking=True))
                if theory_model:
                    feature = net(aug_bank.float())
                # else:
                #     feature, out = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            label_bank = torch.cat(label_bank, dim=0).contiguous()
            feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        feature_bank = feature_bank.cpu().numpy()
        label_bank = label_bank.cpu().numpy()
        output_DBindex = metrics.davies_bouldin_score(feature_bank, label_bank)
        return_values["output_DBindex"] = output_DBindex

    if theory_model:
        input_bank = test_data_loader.dataset.data[:, :, 0, 0]
        if thoery_schedule_dim == 90:
            level_dim = [0, 10, 30, 50, 70, 90]
        elif thoery_schedule_dim == 150:
            level_dim = [0, 10, 30, 60, 100, 150]
        elif thoery_schedule_dim == 20:
            level_dim = [0, 10, 20]
        elif thoery_schedule_dim == 30:
            level_dim = [0, 10, 20, 30]
        elif thoery_schedule_dim == 10:
            level_dim = [0, 10]
        elif thoery_schedule_dim == 50:
            level_dim = [0, 10, 20, 30, 40, 50]
        else:
            raise("Wrong thoery_schedule_dim!")
        level = len(level_dim) - 1
        average_aug_number = 5
        input_DBindex_list = []
        all_dim_input_DBindex_list = []
        gaussian_schedule = [1, 0, 0, 0, 0]
        dup_aug_bank_list = []
        dug_labels = []
        for _ in range(average_aug_number):
            # s_feature_dim = 10
            drop_mask = []
            if theory_aug_by_order:
                # input('check here')
                for i in range(0, level):
                    ids = np.arange(level_dim[i], level_dim[i+1])
                    drop_feature = ids[:random_drop_feature_num[i]]
                    drop_mask.append(drop_feature)
            else:
                for i in range(0, level):
                    ids = np.arange(level_dim[i], level_dim[i+1])
                    random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
                    drop_mask.append(random_frop_feature)
            # input(drop_mask1)
            drop_mask = np.concatenate(drop_mask, axis=0)

            aug = np.ones(input_bank.shape[1])
            for drop_feat in drop_mask:
                aug[drop_feat] = 0.0
            # print(np.sum(aug))
            aug_bank = input_bank * aug
            if gaussian_aug_std != 0:
                gaussian_aug = []
                for i in range(0, level):
                    mean = np.array([0 for _ in range(level_dim[i+1] - level_dim[i])])
                    std = np.diag([gaussian_aug_std * gaussian_schedule[i] for _ in range(level_dim[i+1] - level_dim[i])])
                    aug_noise = np.random.multivariate_normal(mean=mean, cov=std, size=input_bank.shape[0])
                    gaussian_aug.append(aug_noise)
                gaussian_aug = np.concatenate(gaussian_aug, axis=1)
                # gaussian_aug = torch.tensor(gaussian_aug).cuda(non_blocking=True).repeat([data.shape[0], 1])
                aug_bank += gaussian_aug
            dup_aug_bank_list.append(aug_bank)
            dug_labels.append(feature_labels)
            if not instance_level:
                input_DBindex = metrics.davies_bouldin_score(aug_bank[:, dim_range], feature_labels)
                all_dim_input_DBindex = metrics.davies_bouldin_score(aug_bank, feature_labels)
                input_DBindex_list.append(input_DBindex)
                all_dim_input_DBindex_list.append(all_dim_input_DBindex)
        dup_aug_bank = np.concatenate(dup_aug_bank_list, axis=0)
        dug_labels = np.concatenate(dug_labels, axis=0)
        if not instance_level:
            return_values["input_DBindex"] = np.mean(input_DBindex_list)
        return_values["dup_aug_DBindex"] = metrics.davies_bouldin_score(dup_aug_bank[:, dim_range], dug_labels)
        return_values["all_dim_dup_aug_DBindex"] = metrics.davies_bouldin_score(dup_aug_bank, dug_labels)

    return return_values

def test_instance_sim(net, memory_data_loader, test_data_loader, k, temperature, epochs, augmentation, augmentation_prob):
    net.eval()
    total_top1, total_top5, total_num, feature_bank1 = 0.0, 0.0, 0, []
    feature_bank1, feature_bank2, feature_bank, sim_list = [], [], [], []
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      }
    if np.sum(augmentation_prob) == 0:
        if augmentation in transform_func:
            my_transform_func = transform_func[augmentation]
        else:
            raise("Wrong augmentation.")
    else:
        my_transform_func = utils.train_diff_transform_prob(*augmentation_prob)
        
    with torch.no_grad():
        
        feature_bank = []
        posiness = []
        for i in range(3):
            data_iter = iter(test_data_loader)
            end_of_iteration = "END_OF_ITERATION"
            total_top1, total_top5, total_num = 0.0, 0.0, 0.0
            feature_bank1, feature_bank2, = [], []

            for pos_samples_1, pos_samples_2, labels in tqdm(test_data_loader, desc='Feature extracting'):

                pos_samples_1, pos_samples_2, labels = pos_samples_1.cuda(non_blocking=True), pos_samples_2.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                target = torch.arange(0, pos_samples_1.shape[0]).cuda(non_blocking=True)

                net.eval()
                pos_samples_1 = my_transform_func(pos_samples_1)
                pos_samples_2 = my_transform_func(pos_samples_2)
                feature1, out1 = net(pos_samples_1)
                feature2, out2 = net(pos_samples_2)
                feature_bank1.append(feature1)
                feature_bank2.append(feature2)
                
            feature1 = torch.cat(feature_bank1, dim=0).contiguous()
            feature2 = torch.cat(feature_bank2, dim=0).contiguous()
    
            target = torch.arange(0, feature1.shape[0]).cuda(non_blocking=True)
            
            # compute cos similarity between each two groups of augmented samples ---> [B, B]
            sim_matrix = torch.mm(feature1, feature2.t())
            pos_sim = torch.sum(feature1 * feature2, dim=-1)
            
            mask2 = (torch.ones_like(sim_matrix) - torch.eye(feature1.shape[0], device=sim_matrix.device)).bool()
            # [B, B-1]
            neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(feature1.shape[0], -1)
            sim_weight, sim_indices = neg_sim_matrix2.topk(k=1, dim=-1)
            posiness.append(pos_sim - sim_weight.squeeze(1))
            
            sim_indice_1 = sim_matrix.argsort(dim=0, descending=True) #[B, B]
            sim_indice_2 = sim_matrix.argsort(dim=1, descending=True) #[B, B]
            # print(sim_indice_1[0, :30])
            # print(sim_indice_2[:30, 0])

            total_top1 += torch.sum((sim_indice_1[:1, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top1 += torch.sum((sim_indice_2[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_1[:5, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_2[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += feature1.shape[0] * 2
        
        # print(total_top1 / total_num * 100, total_top5 / total_num * 100, )
        posiness = torch.stack(posiness, dim = 1).mean(dim=1)
        easy_weight_50, easy_50 = posiness.topk(k=50, dim=0)
        print(easy_weight_50)
        input(easy_50)

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def test_instance_sim_thoery(net, memory_data_loader, test_data_loader, k, temperature, random_drop_feature_num, thoery_schedule_dim=90):
    net.eval()
    total_top1, total_top5, total_num, feature_bank1 = 0.0, 0.0, 0, []
    feature_bank1, feature_bank2, feature_bank, sim_list = [], [], [], []
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
        
    with torch.no_grad():
        
        feature_bank = []
        posiness = []
        for i in range(3):
            data_iter = iter(test_data_loader)
            end_of_iteration = "END_OF_ITERATION"
            total_top1, total_top5, total_num = 0.0, 0.0, 0.0
            feature_bank1, feature_bank2, = [], []

            for pos_samples_1, pos_samples_2, labels in tqdm(test_data_loader, desc='Feature extracting'):

                pos_samples_1, pos_samples_2, labels = pos_samples_1.cuda(non_blocking=True)[:,0,:,0], pos_samples_2.cuda(non_blocking=True)[:,0,:,0], labels.cuda(non_blocking=True)
                target = torch.arange(0, pos_samples_1.shape[0]).cuda(non_blocking=True)

                net.eval()
                # pos_samples_1 = my_transform_func(pos_samples_1)
                # pos_samples_2 = my_transform_func(pos_samples_2)

                pos_1 = pos_samples_1
                pos_2 = pos_samples_2

                drop_mask1 = []
                drop_mask2 = []
                level = 5
                if thoery_schedule_dim == 90:
                    level_dim = [0, 10, 30, 50, 70, 90]
                elif thoery_schedule_dim == 150:
                    level_dim = [0, 10, 30, 60, 100, 150]
                elif thoery_schedule_dim == 20:
                    level_dim = [0, 10, 20]
                elif thoery_schedule_dim == 30:
                    level_dim = [0, 10, 20, 30]
                elif thoery_schedule_dim == 10:
                    level_dim = [0, 10]
                elif thoery_schedule_dim == 50:
                    level_dim = [0, 10, 20, 30, 40, 50]
                else:
                    raise("Wrong thoery_schedule_dim!")
                level = len(level_dim) - 1
                gaussian_schedule = [10, 4, 2, 1, 0.5]
                # s_feature_dim = 10
                for i in range(0, level):
                    ids = np.arange(level_dim[i], level_dim[i+1])
                    random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
                    drop_mask1.append(random_frop_feature)
                    random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[i]]
                    drop_mask2.append(random_frop_feature)
                # input(drop_mask1)
                drop_mask1 = np.concatenate(drop_mask1, axis=0)
                drop_mask2 = np.concatenate(drop_mask2, axis=0)
                aug1 = np.ones(pos_1.shape[1])
                for drop_feat in drop_mask1:
                    aug1[drop_feat] = 0.0
                aug2 = np.ones(pos_2.shape[1])
                for drop_feat in drop_mask2:
                    aug2[drop_feat] = 0.0
                aug1 = torch.tensor(aug1).cuda(non_blocking=True)
                aug2 = torch.tensor(aug2).cuda(non_blocking=True)
                # input(drop_mask1)
                pos_1 = (pos_1 * aug1).float()
                pos_2 = (pos_2 * aug2).float()

                feature1 = net(pos_1)
                feature2 = net(pos_2)
                feature_bank1.append(feature1)
                feature_bank2.append(feature2)
                
            feature1 = torch.cat(feature_bank1, dim=0).contiguous()
            feature2 = torch.cat(feature_bank2, dim=0).contiguous()
    
            target = torch.arange(0, feature1.shape[0]).cuda(non_blocking=True)
            
            # compute cos similarity between each two groups of augmented samples ---> [B, B]
            sim_matrix = torch.mm(feature1, feature2.t())
            pos_sim = torch.sum(feature1 * feature2, dim=-1)
            
            mask2 = (torch.ones_like(sim_matrix) - torch.eye(feature1.shape[0], device=sim_matrix.device)).bool()
            # [B, B-1]
            neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(feature1.shape[0], -1)
            sim_weight, sim_indices = neg_sim_matrix2.topk(k=1, dim=-1)
            posiness.append(pos_sim - sim_weight.squeeze(1))
            
            sim_indice_1 = sim_matrix.argsort(dim=0, descending=True) #[B, B]
            sim_indice_2 = sim_matrix.argsort(dim=1, descending=True) #[B, B]
            # print(sim_indice_1[0, :30])
            # print(sim_indice_2[:30, 0])

            total_top1 += torch.sum((sim_indice_1[:1, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top1 += torch.sum((sim_indice_2[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_1[:5, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_2[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += feature1.shape[0] * 2
        
        # print(total_top1 / total_num * 100, total_top5 / total_num * 100, )
        posiness = torch.stack(posiness, dim = 1).mean(dim=1)
        easy_weight_50, easy_50 = posiness.topk(k=50, dim=0)

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def test_ssl_softmax(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, logits, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            # print("data.shape:", data.shape)
            # print("feature.shape:", feature.shape)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, logits, out = net(data)

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

# test for one epoch, use noised image to visualize
def test_ssl_visualization(net, test_data_visualization, random_noise_class_test, classwise_noise, pre_load_name, flag_test=False):
    net.eval()
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
    feature_bank = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    with torch.no_grad():
        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
        # generate feature bank
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)
        feature_tsne_input = feature_bank.cpu().numpy().transpose()[:1000]
        labels_tsne_color = feature_labels.cpu().numpy()[:1000]
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("clean data with original label")
        plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=plt.cm.Spectral)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./results/{}_cleandata_orglabel.png'.format(pre_load_name))

        if flag_test:

            test_data_visualization.add_noise_test_visualization(random_noise_class_test, classwise_noise)

            test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
            # generate feature bank
            feature_bank = []
            for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting'):
                feature, out = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            # [N]
            # feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)
            noise_labels = random_noise_class_test[:1000]
            feature_tsne_input = feature_bank.cpu().numpy().transpose()[:1000]
            # labels_tsne_color = feature_labels.cpu().numpy()[:1000]
            feature_tsne_output = tsne.fit_transform(feature_tsne_input)
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.title("noise data with original label")
            plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=plt.cm.Spectral)
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./results/{}_noisedata_orglabel.png'.format(pre_load_name))

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.title("noise data with noise label")
            plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=noise_labels, cmap=plt.cm.Spectral)
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./results/{}_noisedata_noiselabel.png'.format(pre_load_name))

            # for i in range(10):
            #     index_group = np.where(labels_tsne_color==i)
            #     # # labels_tsne_color_group = labels_tsne_color[index_group]
            #     # # noise_labels_group = noise_labels[index_group]
            #     # print(type(feature_tsne_output[index_group, 0]))
            #     # print(type(feature_tsne_output[:,0]))
            #     # print(type(feature_tsne_output[index_group, 1]))
            #     # # print(type(noise_labels_group))
            #     # print(feature_tsne_output[index_group, 0].shape)
            #     # print(feature_tsne_output.shape)
            #     # print(feature_tsne_output[index_group, 1].shape)
            #     # # print(noise_labels_group.shape)

            #     fig = plt.figure(figsize=(8, 8))
            #     ax = fig.add_subplot(1, 1, 1)
            #     plt.title("noise data with original label")
            #     plt.scatter(feature_tsne_output[:,0][index_group], feature_tsne_output[:,1][index_group], s=10, c=labels_tsne_color[index_group], cmap=plt.cm.Spectral)
            #     ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            #     ax.yaxis.set_major_formatter(NullFormatter())
            #     plt.savefig('./visualization/noisedata_orglabel_org{}.png'.format(i))
            #     plt.close()

            #     fig = plt.figure(figsize=(8, 8))
            #     ax = fig.add_subplot(1, 1, 1)
            #     plt.title("noise data with noise label")
            #     plt.scatter(feature_tsne_output[:,0][index_group], feature_tsne_output[:,1][index_group], s=10, c=noise_labels[index_group], cmap=plt.cm.Spectral)
            #     ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            #     ax.yaxis.set_major_formatter(NullFormatter())
            #     plt.savefig('./visualization/noisedata_noiselabel_org{}.png'.format(i))
            #     plt.close()


    return 

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_ssl_for_simclrpy(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

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

print ("__name__", __name__)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    arch = args.arch

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    #                           drop_last=True)
    # train_data = utils.SameImgCIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
    # sys.exit()
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # print(type(train_data))

    # model setup and optimizer config
    model = Model(feature_dim, arch=args.arch).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_ssl_for_simclrpy(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
