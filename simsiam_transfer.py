import argparse
parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--perturbation_budget', default=32.0, type=float, help='perturbation_budget')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
parser.add_argument('--local_dev', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--class_4', action='store_true', default=False)
parser.add_argument('--job_id', default='', type=str, help='The Slurm JOB ID')
parser.add_argument('--pre_load_name', default='', type=str, help='The backbone of encoder')
parser.add_argument('--samplewise', action='store_true', default=False)
parser.add_argument('--orglabel', action='store_true', default=False)
parser.add_argument('--save_img_group', action='store_true', default=False)
parser.add_argument('--perturb_rate', default=1.0, type=float, help='perturbation_rate')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--clean_train', action='store_true', default=False)
parser.add_argument('--pytorch_aug', action='store_true', default=False)
parser.add_argument('--no_bn', action='store_true', default=False)
parser.add_argument('--save_noise_input_space', action='store_true', default=False)
parser.add_argument('--train_data_type', default='CIFAR10', type=str, help='The backbone of encoder')
parser.add_argument('--ignore_model_size', action='store_true', default=False)

parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')

parser.add_argument('--load_piermaro_model', action='store_true', default=False)
parser.add_argument('--load_piermaro_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--piermaro_whole_epoch', default='', type=str, help='Whole epoch when use re_job to train')
parser.add_argument('--piermaro_restart_epoch', default=0, type=int, help='The order of epoch when use re_job to train')

# args parse
args = parser.parse_args()
import os
if args.local_dev != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local_dev
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
from utils import train_diff_transform, train_transform_no_totensor
import datetime

from resnet_big_normal import Model_bn
from resnet_big import Model_no_bn
from resnet_cifar import resnet18_test

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

from simsiam_utils import train_simsiam, test_simsiam
# from simsiam_pytorch import BYOL
from methods import set_model
from torchvision import models

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("check check")

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        if not args.pytorch_aug:
            raise('Please use pytorch_aug')
            pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)
        
        batch_train_loss, batch_size_count = train_simsiam(model, pos_1, pos_2, train_optimizer, args.batch_size, args.temperature, noise_after_transform=False)

        total_num += batch_size_count
        total_loss += batch_train_loss
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

def train_simclr_noise_return_loss_tensor(net, pos_1, pos_2, train_optimizer, batch_size, temperature):
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

def train_simclr_noise_pos1_pertub(net, pos_1, pos_2, train_optimizer, batch_size, temperature):
    # train a batch
    # print(pos_1.shape)
    # print(pos_2.shape)
    # print("batch_size", batch_size)
    # print("this is train_simclr_noise")
    # pos_1 = torch.clamp(pos_samples_1.data, 0, 1)
    # pos_2 = torch.clamp(pos_samples_2.data, 0, 1)
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
    train_optimizer.zero_grad()
    pos_1.retain_grad()
    loss.backward()
    # train_optimizer.step() # step is used to update Variable. But we do not update like SGD. We use the sign of gradients

    # total_num += batch_size
    total_loss = loss.item()
    # train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / pos_1.shape[0]

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test_ssl(net, memory_data_loader, test_data_loader, k, temperature, epoch, epochs):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    # c = 10
    c = np.max(memory_data_loader.dataset.targets) + 1
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
def test_ssl_visualization(net, test_data_visualization, random_noise_class_test, classwise_noise):
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
        plt.savefig('./visualization/cleandata_orglabel.png')

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
        plt.savefig('./visualization/noisedata_orglabel.png')

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("noise data with noise label")
        plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=noise_labels, cmap=plt.cm.Spectral)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./visualization/noisedata_noiselabel.png')

        for i in range(10):
            index_group = np.where(labels_tsne_color==i)
            # # labels_tsne_color_group = labels_tsne_color[index_group]
            # # noise_labels_group = noise_labels[index_group]
            # print(type(feature_tsne_output[index_group, 0]))
            # print(type(feature_tsne_output[:,0]))
            # print(type(feature_tsne_output[index_group, 1]))
            # # print(type(noise_labels_group))
            # print(feature_tsne_output[index_group, 0].shape)
            # print(feature_tsne_output.shape)
            # print(feature_tsne_output[index_group, 1].shape)
            # # print(noise_labels_group.shape)

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.title("noise data with original label")
            plt.scatter(feature_tsne_output[:,0][index_group], feature_tsne_output[:,1][index_group], s=10, c=labels_tsne_color[index_group], cmap=plt.cm.Spectral)
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./visualization/noisedata_orglabel_org{}.png'.format(i))
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.title("noise data with noise label")
            plt.scatter(feature_tsne_output[:,0][index_group], feature_tsne_output[:,1][index_group], s=10, c=noise_labels[index_group], cmap=plt.cm.Spectral)
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./visualization/noisedata_noiselabel_org{}.png'.format(i))
            plt.close()

    return 

# for simclrpy, removed some positional parameters, will use global variables to replace them.
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
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    arch = args.arch

    # data prepare
    # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    #                           drop_last=True)
    # unlearnable_samplewise_107535314_1_20211120061615_0.5_512_1000perturbation

    if args.pre_load_name == '':
        raise("Use pre_load_name.")
    else:
        pre_load_name = args.pre_load_name
    
    class_4 = args.class_4
    perturbation_budget = args.perturbation_budget
    samplewise_perturb = args.samplewise
    if class_4:
        random_noise_class_path = 'noise_class_label_1024_4class.npy'
        save_name_pre = pre_load_name + "_budget{}_class4".format(perturbation_budget)
    else:
        random_noise_class_path = 'noise_class_label.npy'
        save_name_pre = pre_load_name + "_budget{}_class10".format(perturbation_budget)

    if args.orglabel:
        save_name_pre += '_orglabel'
    print(save_name_pre)

    if args.train_data_type == 'CIFAR10':
        if args.pytorch_aug:
            train_data = utils.TransferCIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True, perturb_tensor_filepath="./results/{}.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, class_4=class_4, samplewise_perturb=samplewise_perturb, org_label_flag=args.orglabel, flag_save_img_group=args.save_img_group, perturb_rate=args.perturb_rate, clean_train=args.clean_train)
        else:
            train_data = utils.TransferCIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, perturb_tensor_filepath="./results/{}.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, class_4=class_4, samplewise_perturb=samplewise_perturb, org_label_flag=args.orglabel, flag_save_img_group=args.save_img_group, perturb_rate=args.perturb_rate, clean_train=args.clean_train)

        if args.save_img_group:
            train_data.save_noise_img()
        if args.save_noise_input_space:
            noise_input_space = train_data.perturb_tensor.reshape((train_data.perturb_tensor.shape[0], -1))
            utils.plot_feature(noise_input_space[:1000], train_data.targets[:1000], pre_load_name)
            input('save_noise_input_space done')
        # train_data = utils.TransferCIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, perturb_tensor_filepath="./results/{}_checkpoint_perturbation.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, class_4=class_4)
        # load noise here:
        # pretrained_classwise_noise = torch.load("./results/{}_checkpoint_perturbation.pt".format(pre_load_name))
        # random_noise_class = np.load('noise_class_label_1024_4class.npy')
        # train_data.make_unlearnable(random_noise_class, pretrained_classwise_noise)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        # sys.exit()
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=class_4)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, class_4=class_4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif args.train_data_type == 'CIFAR100':
        if args.pytorch_aug:
            train_data = utils.TransferCIFAR100Pair(root='data', train=True, transform=utils.train_transform, download=True, perturb_tensor_filepath="./results/{}.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, samplewise_perturb=samplewise_perturb, org_label_flag=args.orglabel, flag_save_img_group=args.save_img_group, perturb_rate=args.perturb_rate, clean_train=args.clean_train)
        else:
            train_data = utils.TransferCIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, perturb_tensor_filepath="./results/{}.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, samplewise_perturb=samplewise_perturb, org_label_flag=args.orglabel, flag_save_img_group=args.save_img_group, perturb_rate=args.perturb_rate, clean_train=args.clean_train)

        if args.save_img_group:
            train_data.save_noise_img()
        if args.save_noise_input_space:
            noise_input_space = train_data.perturb_tensor.reshape((train_data.perturb_tensor.shape[0], -1))
            utils.plot_feature(noise_input_space[:10000], train_data.targets[:10000], pre_load_name)
            input('save_noise_input_space done')

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        # sys.exit()
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # print(type(train_data))

    # model setup and optimizer config
    # model = Model(feature_dim, arch=args.arch).cuda()
    # model = BYOL(resnet18_test(), image_size = 32, hidden_layer = 'avgpool', use_momentum = True).cuda()
    model = set_model('resnet18', 'cifar10')
    model = model.cuda()

    if args.load_model:
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        try:
            model.load_state_dict(checkpoints['state_dict'])
        except:
            model.load_state_dict(checkpoints)
        logger.info("File %s loaded!" % (load_model_path))

    if args.load_piermaro_model:
        load_model_path = './results/{}.pth'.format(args.load_piermaro_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        model.load_state_dict(checkpoints['state_dict'])

    # if not args.ignore_model_size:
    #     flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),torch.randn(1, 3, 32, 32).cuda()))
    #     flops, params = clever_format([flops, params])
    #     print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.06, weight_decay=0.0005, momentum=0.9)
    optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=0.9, weight_decay=5e-4)

    if args.load_model or args.load_piermaro_model:
        if 'optimizer' in checkpoints:
            optimizer.load_state_dict(checkpoints['optimizer'])

    c = len(memory_data.classes)

    # training loop
    save_name_pre = '{}_retrain_model'.format(save_name_pre)
    if args.piermaro_whole_epoch != '':
        if args.load_piermaro_model:
            save_name_pre = args.load_piermaro_model_path
            save_name_pre = save_name_pre.replace("_piermaro_model", "").replace("_model", "")
        else:
            save_name_pre = 'transfer_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if args.load_piermaro_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
        best_acc = results['best_acc'][len(results['best_acc'])-1]
        best_acc_loss = results['best_acc_loss'][len(results['best_acc_loss'])-1]
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc': [], 'best_acc_loss': []}
        best_acc = 0.0
        best_acc_loss = 10
    if not os.path.exists('results'):
        os.mkdir('results')
    test_acc_1, test_acc_5 = test_simsiam(model.backbone, memory_loader, test_loader, k, temperature, 0, epochs)
    for _epoch in range(1, epochs + 1):
        epoch = _epoch + args.piermaro_restart_epoch
        train_loss = train(model, train_loader, optimizer)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_simsiam(model.backbone, memory_loader, test_loader, k, temperature, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            best_acc_loss = train_loss
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

        if epoch % 10 == 0:
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))

        results['best_acc'].append(best_acc)
        results['best_acc_loss'].append(best_acc_loss)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

        piermaro_checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(piermaro_checkpoint, 'results/{}_piermaro_model.pth'.format(save_name_pre))