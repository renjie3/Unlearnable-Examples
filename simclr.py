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
from utils import train_diff_transform, train_diff_transform2, train_diff_transform_resize48, train_diff_transform_resize64, train_diff_transform_resize28, train_diff_transform_ReCrop_Hflip, train_diff_transform_ReCrop_Hflip_Bri, train_diff_transform_ReCrop_Hflip_Con, train_diff_transform_ReCrop_Hflip_Sat, train_diff_transform_ReCrop_Hflip_Hue

import kornia.augmentation as Kaug
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

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

def train_looc(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], n_zspace=3):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # the order of faugmentation_prob in Z space is [colorJitter, rotation, reCrop, None]
    if n_zspace == 2:
        base_transform = nn.Sequential(Kaug.RandomResizedCrop([32,32]), Kaug.RandomHorizontalFlip(p=0.5), Kaug.RandomGrayscale(p=0.2))
        z_augs = [Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=augmentation_prob[0]), 
                  Kaug.RandomRotation(360, p=augmentation_prob[1])]
    elif n_zspace == 3:
        base_transform = nn.Sequential(Kaug.RandomHorizontalFlip(p=0.5), Kaug.RandomGrayscale(p=0.2)) # probability may influence rotation
        z_augs = [Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=augmentation_prob[0]), 
                  Kaug.RandomRotation(360, p=augmentation_prob[1]),
                  Kaug.RandomResizedCrop([32,32], p=augmentation_prob[2])
                  ]
        
    pos_1 = pos_1.cuda(non_blocking=True)
    view_q = base_transform(pos_1)
    q_aug_params = []
    for z_aug in z_augs:
        view_q = z_aug(view_q)
        q_aug_params.append(z_aug._params)
    
    view_0 = base_transform(pos_1)
    for z_aug in z_augs:
        view_0 = z_aug(view_0)
        
    views = [pos_1, view_q, view_0]
        
    for i in range(n_zspace):
        view = base_transform(pos_1)
        for j in range(len(z_augs)):
            if j == i:
                # example: transform(x_rgb, params=transform._params)
                view = z_augs[j](view, params=q_aug_params[j])
            else:
                view = z_augs[j](view)
        views.append(view)
    
    features, outs = net(views) # [x,q,I0-In]
    z_loss = 0
    z0 = outs[0]
    pos_z0 = 0
    for i in range(2,n_zspace + 3):
        # [B]
        pos_z0 += torch.exp(torch.sum(z0[1] * z0[i], dim=-1) / temperature)
    # [B, B]
    simmat_q_x = torch.exp(torch.mm(z0[1], z0[0].t().contiguous()) / temperature)
    mask = (torch.ones_like(simmat_q_x) - torch.eye(pos_1.shape[0], device=simmat_q_x.device)).bool()
    # [B, B-1]
    simmat_q_x = simmat_q_x.masked_select(mask).view(pos_1.shape[0], -1)
    z_loss += (- torch.log(pos_z0 / simmat_q_x.sum(dim=-1))).mean()
    
    for i in range(3, n_zspace+3):
        zi = outs[i-2]
        pos_zi = 0
        neg_zi_Ik = 0
        for j in range(2,n_zspace+3):
            # [B]
            if j == i:
                pos_zi += torch.exp(torch.sum(zi[1] * zi[j], dim=-1) / temperature)
            else:
                neg_zi_Ik += torch.exp(torch.sum(zi[1] * zi[j], dim=-1) / temperature)
        simmat_q_x = torch.exp(torch.mm(z0[1], z0[0].t().contiguous()) / temperature)
        mask = (torch.ones_like(simmat_q_x) - torch.eye(pos_1.shape[0], device=simmat_q_x.device)).bool()
        # [B, B-1]
        simmat_q_x = simmat_q_x.masked_select(mask).view(pos_1.shape[0], -1)
        z_loss += (- torch.log(pos_zi / (neg_zi_Ik + simmat_q_x.sum(dim=-1)))).mean()
    
    # # [2*B, D]
    # out = torch.cat([q_1, out_2], dim=0)
    # # [2*B, 2*B]
    # sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    # mask = (torch.ones_like(sim_matrix) - torch.eye(2 * pos_1.shape[0], device=sim_matrix.device)).bool()
    # # [2*B, 2*B-1]
    # sim_matrix = sim_matrix.masked_select(mask).view(2 * pos_1.shape[0], -1)

    # # compute loss
    # pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # # [2*B]
    # pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    # loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    
    loss = z_loss
    
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

def train_micl(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0], n_zspace=3):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # the order of faugmentation_prob in Z space is [colorJitter, rotation, reCrop, None]
    # the order of input params of train_diff_transform_prob2 is [recrop, hflip, cj, gray, rot]
    if n_zspace == 2:
        augs = [utils.train_diff_transform_prob2(1.0, 0.5, augmentation_prob[0], 0.2, 0), 
                utils.train_diff_transform_prob2(1.0, 0.5, 0, 0.2, augmentation_prob[1])]
    elif n_zspace == 3:
        augs = [utils.train_diff_transform_prob2(0, 0.5, augmentation_prob[0], 0.2, 0), 
                utils.train_diff_transform_prob2(0, 0.5, 0, 0.2, augmentation_prob[1]),
                utils.train_diff_transform_prob2(augmentation_prob[2], 0.5, 0, 0.2, 0),]
        
    pos_1 = pos_1.cuda(non_blocking=True)
    views_1 = []
    views_2 = []
    for i in range(n_zspace):
        views_1.append(augs[i](pos_1))
        views_2.append(augs[i](pos_1))
    
    features_1, outs_1 = net(views_1)
    features_2, outs_2 = net(views_2)
    
    z_loss = []
    loss = 0
    for i in range(n_zspace):
        out_1, out_2 = outs_1[i], outs_2[i]
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
        loss_single_space = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        loss += loss_single_space
        z_loss.append(loss_single_space)
    
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    total_loss = loss.item()
    for i in range(len(z_loss)):
        z_loss[i] = z_loss[i].item()*pos_1.shape[0]

    return total_loss * pos_1.shape[0], pos_1.shape[0], z_loss

def train_align(net, pos_1, pos_2, train_optimizer, normalize=False):
    net.train()
    total_loss, total_num = 0.0, 0.0
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True)[:,0,:,0], pos_2.cuda(non_blocking=True)[:,0,:,0]
    
    out_1 = net(pos_1)
    out_2 = net(pos_2)
    
    a = 0.01
    
    if normalize:
        l_align = (out_1 - out_2)**2
        loss = torch.mean(l_align)
    else:
        l_align = (out_1 - out_2)**2
        # l_norm = out_1**2 + out_2**2
        loss = torch.mean(l_align) # + a / torch.mean(out_1**2) + a / torch.mean(out_2**2)
        L1_reg = torch.tensor(0., requires_grad=True)
        for name, param in net.named_parameters():
            if 'weight' in name:
                L1_reg = L1_reg + torch.norm(param, 1)
        loss += a / L1_reg
    
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    total_loss = loss.item()

    return total_loss * pos_1.shape[0], pos_1.shape[0]

def train_simclr(net, pos_1, pos_2, train_optimizer, batch_size, temperature, noise_after_transform=False, mix="no", augmentation="simclr", augmentation_prob=[0,0,0,0]):
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

def train_simclr_theory(net, pos_1, pos_2, train_optimizer, batch_size, temperature, random_drop_feature_num):
    net.train()
    total_loss, total_num = 0.0, 0
        
    pos_1, pos_2 = pos_1.cuda(non_blocking=True)[:,0,:,0], pos_2.cuda(non_blocking=True)[:,0,:,0]
    
    drop_mask1 = []
    drop_mask2 = []
    level = 5
    level_dim = 20
    s_feature_dim = 10
    ids = np.arange(0, s_feature_dim)
    random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[0]]
    drop_mask1.append(random_frop_feature)
    random_frop_feature = np.random.permutation(ids)[:random_drop_feature_num[0]]
    drop_mask2.append(random_frop_feature)
    for i in range(1, level):
        ids = np.arange(s_feature_dim+(i-1)*level_dim, s_feature_dim+i*level_dim)
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
    pos_1 = (pos_1 * aug1).float()
    pos_2 = (pos_2 * aug2).float()

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

def train_simclr_noise_return_loss_tensor(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True, noise_after_transform=False):
    net.eval()
    total_loss, total_num = 0.0, 0
    
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    # if flag_strong_aug:
    #     pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
    # else:
    #     pos_1, pos_2 = train_diff_transform2(pos_1), train_diff_transform2(pos_2)
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
def test_ssl(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs, c=10):
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

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def test_intra_inter_sim(net, memory_data_loader, test_data_loader, k, temperature, epochs, distance=False):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    c = 10
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(test_data_loader, desc='Feature extracting'):
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

def test_instance_sim(net, memory_data_loader, test_data_loader, k, temperature, epochs, augmentation, augmentation_prob):
    net.eval()
    total_top1, total_top5, total_num, feature_bank1 = 0.0, 0.0, 0, []
    feature_bank1, feature_bank2, feature_bank, sim_list = [], [], [], []
    c = 10
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

def test_ssl_softmax(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    c = 10
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
    c = 10
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
    c = 10
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
