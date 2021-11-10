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
from utils import train_diff_transform, train_diff_transform2

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

def train_simclr(net, pos_1, pos_2, train_optimizer, batch_size, temperature):
    # train a batch
    # print("pos_1.shape: ", pos_1.shape)
    # print("pos_2.shape: ", pos_2.shape)
    net.train()
    total_loss, total_num = 0.0, 0
    # for pos_1, pos_2, target in train_bar:
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
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
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    # total_num += batch_size
    total_loss = loss.item()
    # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss * pos_1.shape[0], pos_1.shape[0]

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

def train_simclr_noise_return_loss_tensor(net, pos_1, pos_2, train_optimizer, batch_size, temperature, flag_strong_aug = True):
    # train a batch
    # print(pos_1.shape)
    # print(pos_2.shape)
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
def test_ssl(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    c = 10
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
