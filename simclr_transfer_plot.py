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
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='The backbone of encoder')
parser.add_argument('--kmeans_index', default=-1, type=int, help='perturbation_rate')
parser.add_argument('--unlearnable_kmeans_label', action='store_true', default=False)
parser.add_argument('--kmeans_label_file', default='', type=str, help='The backbone of encoder')

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets


if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]

else:
    device = torch.device('cpu')

print("check check")

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        if args.pytorch_aug:
            pos_1, pos_2 = train_transform_no_totensor(pos_1), train_transform_no_totensor(pos_2)
            print(pos_1)
            input()
        else:
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

# test for one epoch, use noised image to visualize
def test_ssl_visualization(net, test_data_visualization):
    net.eval()
    c = 10
    feature_bank = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    with torch.no_grad():
        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        # generate feature bank
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            if len(feature_bank) >= 2:
                break
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)
        feature_tsne_input = feature_bank.cpu().numpy().transpose()[:1024]
        labels_tsne_color = feature_labels.cpu().numpy()[:1024]
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("clean data with original label")
        cm = plt.cm.get_cmap('gist_rainbow', c)
        plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=cm)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./visualization/cleandata_orglabel.png')

    return 

def test_ssl_visualization_noise(perturb, labels):
    c = 10
    feature_bank = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # [D, N]
    print(perturb.shape)
    feature_bank = perturb.view(1024, -1)
    print(feature_bank.shape)
    # [N]
    feature_tsne_input = feature_bank.detach().cpu().numpy()[:1024]
    labels_tsne_color = labels[:1024]
    feature_tsne_output = tsne.fit_transform(feature_tsne_input)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.title("clean data with original label")
    plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.savefig('./visualization/noise.png')

    return 

def test_ssl_visualization_noise2(net, perturb, labels):
    net.eval()
    c = 10
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # [D, N]
    
    feature_bank, out_bank = net(perturb.cuda())

    # [N]
    feature_tsne_input = feature_bank.detach().cpu().numpy()[:1024]
    labels_tsne_color = labels[:1024]
    feature_tsne_output = tsne.fit_transform(feature_tsne_input)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.title("clean data with original label")
    plt.scatter(feature_tsne_output[:, 0], feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=plt.cm.Spectral)
    ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.savefig('./visualization/noise2.png')

    return 

# for simclrpy, removed some positional parameters, will use global variables to replace them.
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
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    arch = args.arch

    # data prepare
    # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    #                           drop_last=True)
    # unlearnable_samplewise_107535314_1_20211120061615_0.5_512_1000perturbation

    if args.pre_load_name == '':
        # raise("Use pre_load_name.")
        pass
    else:
        pre_load_name = args.pre_load_name
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

    if args.pre_load_name != '':
        perturb_tensor_filepath = "./results/{}.pt".format(pre_load_name)
    else:
        perturb_tensor_filepath = None

    train_data = utils.TransferCIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, perturb_tensor_filepath=perturb_tensor_filepath, random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, class_4=class_4, samplewise_perturb=samplewise_perturb, org_label_flag=args.orglabel, flag_save_img_group=args.save_img_group, perturb_rate=args.perturb_rate, clean_train=args.clean_train, kmeans_index=args.kmeans_index, unlearnable_kmeans_label=args.unlearnable_kmeans_label, kmeans_label_file=args.kmeans_label_file)
    if args.save_img_group:
        train_data.save_noise_img()
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

    # print(type(train_data))

    # model setup and optimizer config
    model = Model(feature_dim, arch=args.arch).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1, weight_decay=0.0005, momentum=0.9)
    c = len(memory_data.classes)

    if args.load_model:
        # unlearnable_cleantrain_41501264_1_20211204151414_0.5_512_1000_final_model
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        filter_name_checkpoints = {}
        for key in checkpoints:
            filter_name_checkpoints[key.replace('module.', '')] = checkpoints[key]
        model.load_state_dict(filter_name_checkpoints)


    epoch=0
    # test_acc_1, test_acc_5 = test_ssl_for_simclrpy(model, memory_loader, test_loader)

    test_ssl_visualization(model, train_data)
    # print(type(train_data.perturb_tensor))
    # print(train_data.perturb_tensor.shape)
    # test_ssl_visualization_noise2(model, train_data.perturb_tensor, train_data.targets)
    # perturb_tensor

    # training loop
    # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc': [], 'best_acc_loss': []}
    # save_name_pre = '{}_retrain_model'.format(save_name_pre)
    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # best_acc = 0.0
    # best_acc_loss = 10
    # for epoch in range(1, epochs + 1):
    #     train_loss = train(model, train_loader, optimizer)
    #     results['train_loss'].append(train_loss)
    #     test_acc_1, test_acc_5 = test_ssl_for_simclrpy(model, memory_loader, test_loader)
    #     results['test_acc@1'].append(test_acc_1)
    #     results['test_acc@5'].append(test_acc_5)
    #     if test_acc_1 > best_acc:
    #         best_acc = test_acc_1
    #         best_acc_loss = train_loss
    #         if not args.no_save:
    #             torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    #     results['best_acc'].append(best_acc)
    #     results['best_acc_loss'].append(best_acc_loss)

    #     # save statistics
    #     data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    #     if not args.no_save:
    #         data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
