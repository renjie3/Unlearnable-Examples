import argparse
# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='../datasets')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.2, type=float)
parser.add_argument('--universal_stop_error', default=0.5, type=float)
parser.add_argument('--universal_train_target', default='train_subset', type=str)
parser.add_argument('--train_step', default=10, type=int)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-max', 'random'], help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='noise shape')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)
# Self-supervised Options
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
parser.add_argument('--local_dev', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--noise_num', default='10', type=int, help='The number of categories of misleading noise')
parser.add_argument('--model_parameters_path', default='./my_model_parameters', type=str, help='The path to save model parameters')
args = parser.parse_args()

import collections
import datetime
import os
if args.local_dev != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local_dev
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import util
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer
import sys

import utils
import datetime
from model import Model
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import test_ssl, train_simclr, test_ssl_visualization
mlconfig.register(madrys.MadrysLoss)
from thop import profile, clever_format

import matplotlib.pyplot as plt
import matplotlib
from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

# Convert Eps
args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255

# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def plot_distribution(net, test_data_visualization, samplewise_noise, pre_load_name):
    net.eval()
    # c = 10
    c = np.max(test_data_visualization.targets) + 1
    feature_bank = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    with torch.no_grad():
        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
        # generate feature bank
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting on org images'):
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

        test_data_visualization.add_samplewise_noise_test_visualization(samplewise_noise)

        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
        # generate feature bank
        perturbed_feature_bank = []
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting on perturbed images'):
            feature, out = net(data.cuda(non_blocking=True))
            perturbed_feature_bank.append(feature)
        # [D, N]
        perturbed_feature_bank = torch.cat(perturbed_feature_bank, dim=0).t().contiguous()
        # [N]
        # feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)
        perturbed_feature_tsne_input = perturbed_feature_bank.cpu().numpy().transpose()[:1000]
        # labels_tsne_color = feature_labels.cpu().numpy()[:1000]
        perturbed_feature_tsne_output = tsne.fit_transform(perturbed_feature_tsne_input)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("noise data with original label")
        plt.scatter(perturbed_feature_tsne_output[:, 0], perturbed_feature_tsne_output[:, 1], s=10, c=labels_tsne_color, cmap=plt.cm.Spectral)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./results/{}_noisedata_orglabel.png'.format(pre_load_name))

def plot_distribution_2D(net, test_data_visualization, samplewise_noise, pre_load_name):
    net.eval()
    # c = 10
    c = np.max(test_data_visualization.targets) + 1
    out_bank = []
    with torch.no_grad():
        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
        # generate feature bank
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting on org images'):
            feature, out = net(data.cuda(non_blocking=True))
            out_bank.append(out)
        # [D, N]
        out_bank = torch.cat(out_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=out_bank.device)
        out_plot = out_bank.cpu().numpy().transpose()[:1000]
        labels_color = feature_labels.cpu().numpy()[:1000]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("clean data with original label")
        plt.scatter(out_plot[:, 0], out_plot[:, 1], s=10, c=labels_color, cmap=plt.cm.Spectral)
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./results/{}_cleandata_orglabel.png'.format(pre_load_name))
        plt.close()

        color_list = ['r', 'g', 'b', 'y']

        fig = plt.figure(figsize=(8, 8))
        for i in range(4):
            ax = fig.add_subplot(2, 2, i+1)
            plt.title("clean data with class {} original label".format(i))
            one_class_index = np.where(labels_color == i)
            # print(one_class_index[0].shape)
            # print(type(one_class_index[0]))
            plt.scatter(out_plot[one_class_index[0], 0], out_plot[one_class_index[0], 1], s=1, c=color_list[i], cmap=plt.cm.Spectral)
            plt.xlim((-1.2, 1.2))
            plt.ylim((-1.2, 1.2))
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./results/{}_one_class_cleandata_orglabel.png'.format(pre_load_name))
        plt.close()

        if samplewise_noise != None: 

            test_data_visualization.add_samplewise_noise_test_visualization(samplewise_noise)

            test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
            # generate feature bank
            perturbed_out_bank = []
            for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting on perturbed images'):
                feature, out = net(data.cuda(non_blocking=True))
                perturbed_out_bank.append(out)
            # [D, N]
            perturbed_out_bank = torch.cat(perturbed_out_bank, dim=0).t().contiguous()
            # [N]
            # feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)
            perturbed_out_plot = perturbed_out_bank.cpu().numpy().transpose()[:1000]
            # labels_tsne_color = feature_labels.cpu().numpy()[:1000]
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1)
            plt.title("noise data with original label")
            plt.scatter(perturbed_out_plot[:, 0], perturbed_out_plot[:, 1], s=10, c=labels_color, cmap=plt.cm.Spectral)
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./results/{}_noisedata_orglabel.png'.format(pre_load_name))
            plt.close()
            fig = plt.figure(figsize=(8, 8))
            for i in range(4):
                ax = fig.add_subplot(2, 2, i+1)
                plt.title("noise data with class {} original label".format(i))
                one_class_index = np.where(labels_color == i)
                # print(labels_color.shape)
                plt.scatter(perturbed_out_plot[one_class_index[0], 0], perturbed_out_plot[one_class_index[0], 1], s=1, c=color_list[i], cmap=plt.cm.Spectral)
                plt.xlim((-1.2, 1.2))
                plt.ylim((-1.2, 1.2))
                ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
                ax.yaxis.set_major_formatter(NullFormatter())
            plt.savefig('./results/{}_one_class_noisedata_orglabel.png'.format(pre_load_name))
            plt.close()

def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    # ssl does not use this
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100

        # Reset Stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)
    return


def universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    random_noise = random_noise.to(device)
    model = model.to(device)
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i in range(len(labels)):
                class_index = labels[i].item()
                noise = random_noise[class_index]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=images[i].shape, patch_location=args.patch_location)
                images[i] += class_noise
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k):
    # Class-Wise perturbation
    # Generate Data loader

    # # training loop
    # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    # save_name_pre = 'unlearnable_{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # best_acc = 0.0
    # for epoch in range(1, epochs + 1):
    #     train_loss = train_simclr(model, train_loader, optimizer)
    #     results['train_loss'].append(train_loss)
    #     test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader)
    #     results['test_acc@1'].append(test_acc_1)
    #     results['test_acc@5'].append(test_acc_5)
    #     # save statistics
    #     data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    #     data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
    #     if test_acc_1 > best_acc:
    #         best_acc = test_acc_1
    #         torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

    condition = True
    # data_iter = iter(data_loader['train_dataset'])
    epochs = 3
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = 'unlearnable_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    for epoch_idx in range(1, epochs+1):
        # print(epoch_idx, condition)
        # for item in train_loader_simclr:
            # print("item shape: ", item[0].shape)
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0,0
        # logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
        condition = True
        if hasattr(model, 'classify'):
            model.classify = True
        while condition:
            if args.attack_type == 'min-min' and not args.load_model:
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    try:
                        next_item = next(data_iter, end_of_iteration)
                        if next_item != end_of_iteration:
                            (pos_samples_1, pos_samples_2, labels) = next_item
                            
                        else:
                            condition = False
                            del data_iter
                            break
                    except:
                        # data_iter = iter(data_loader['train_dataset'])
                        # (pos_1, pos_2, labels) = next(data_iter)
                        raise('train loader iteration problem')
                        
                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    # Add Class-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        noise = random_noise[label.item()]
                        mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_1.shape, patch_location=args.patch_location)
                        train_pos_1.append(pos_samples_1[i]+class_noise)
                        train_pos_2.append(pos_samples_2[i]+class_noise)
                    # Train
                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    # trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)
                    batch_train_loss, batch_size_count = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature)
                    sum_train_loss += batch_train_loss * batch_size_count
                    sum_train_batch_size += batch_size_count
                
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)
                # Add Class-wise Noise to each sample
                batch_noise, mask_cord_list = [], []
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    noise = random_noise[label.item()]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_1.shape, patch_location=args.patch_location)
                    batch_noise.append(class_noise)
                    mask_cord_list.append(mask_cord)

                # Update universal perturbation
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                batch_noise = torch.stack(batch_noise).to(device)
                if args.attack_type == 'min-min':
                    perturb_img, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_print(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature)
                    train_noise_loss_sum += train_noise_loss * pos_samples_1.shape[0]
                    train_noise_loss_count += pos_samples_1.shape[0]
                    # perturb_img, eta = noise_generator.min_min_attack_simclr(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature)
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                class_noise_eta = collections.defaultdict(list)
                for i in range(len(eta)):
                    x1, x2, y1, y2 = mask_cord_list[i]
                    delta = eta[i][:, x1: x2, y1: y2]
                    class_noise_eta[labels[i].item()].append(delta.detach().cpu())

                for key in class_noise_eta:
                    delta = torch.stack(class_noise_eta[key]).mean(dim=0) - random_noise[key]
                    class_noise = random_noise[key]
                    class_noise += delta
                    random_noise[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon) # important.
                # print(random_noise)
            # print(train_noise_loss_sum / float(train_noise_loss_count))

            # # Eval termination conditions
            # loss_avg, error_rate = universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=args.universal_train_target)
            # logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))
            # random_noise = random_noise.detach()
            # ENV['random_noise'] = random_noise
            # if args.attack_type == 'min-min':
            #     condition = error_rate > args.universal_stop_error
            # elif args.attack_type == 'min-max':
            #     condition = error_rate < args.universal_stop_error
        
        train_loss = sum_train_loss / float(sum_train_batch_size)
        print(train_loss)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
    print('save at results/{}_final_model.pth'.format(save_name_pre))

    return random_noise, save_name_pre


def samplewise_perturbation_eval(random_noise, data_loader, model, eval_target='train_dataset', mask_cord_list=[]):
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    # random_noise = random_noise.to(device)
    model = model.to(device)
    idx = 0
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if random_noise is not None:
            for i, (image, label) in enumerate(zip(images, labels)):
                if not torch.is_tensor(random_noise):
                    sample_noise = torch.tensor(random_noise[idx]).to(device)
                else:
                    sample_noise = random_noise[idx].to(device)
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                images[i] = images[i] + sample_noise
                idx += 1
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
    return loss_meter.avg, err_meter.avg


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV):
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed, no_train_augments=True)

    if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
        data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
        data_loader['train_dataset'] = data_loader['train_subset']
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    mask_cord_list = []
    idx = 0
    for images, labels in data_loader['train_dataset']:
        for i, (image, label) in enumerate(zip(images, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    condition = True
    train_idx = 0
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    while condition:
        if args.attack_type == 'min-min' and not args.load_model:
            # Train Batch for min-min noise
            for j in tqdm(range(0, args.train_step), total=args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    train_idx = 0
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images, labels = images.to(device), labels.to(device)
                # Add Sample-wise Noise to each sample
                for i, (image, label) in enumerate(zip(images, labels)):
                    sample_noise = random_noise[train_idx]
                    c, h, w = image.shape[0], image.shape[1], image.shape[2]
                    mask = np.zeros((c, h, w), np.float32)
                    x1, x2, y1, y2 = mask_cord_list[train_idx]
                    if type(sample_noise) is np.ndarray:
                        mask[:, x1: x2, y1: y2] = sample_noise
                    else:
                        mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    sample_noise = torch.from_numpy(mask).to(device)
                    images[i] = images[i] + sample_noise
                    train_idx += 1

                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(images, labels, model, optimizer)

        # Search For Noise
        idx = 0
        for i, (images, labels) in tqdm(enumerate(data_loader['train_dataset']), total=len(data_loader['train_dataset'])):
            images, labels, model = images.to(device), labels.to(device), model.to(device)

            # Add Sample-wise Noise to each sample
            batch_noise, batch_start_idx = [], idx
            for i, (image, label) in enumerate(zip(images, labels)):
                sample_noise = random_noise[idx]
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                if type(sample_noise) is np.ndarray:
                    mask[:, x1: x2, y1: y2] = sample_noise
                else:
                    mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                batch_noise.append(sample_noise)
                idx += 1

            # Update sample-wise perturbation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')

            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                if torch.is_tensor(random_noise):
                    random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

        # Eval termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(random_noise, data_loader, model, eval_target='train_dataset',
                                                            mask_cord_list=mask_cord_list)
        logger.info('Loss: {:.4f} Acc: {:.2f}%'.format(loss_avg, 100 - error_rate*100))

        if torch.is_tensor(random_noise):
            random_noise = random_noise.detach()
            ENV['random_noise'] = random_noise
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = image.shape[0], image.shape[1], image.shape[2]
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise
    else:
        return random_noise


def main():
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    arch = args.arch
    # Setup ENV
    # datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
    #                                               eval_batch_size=args.eval_batch_size,
    #                                               train_data_type=args.train_data_type,
    #                                               train_data_path=args.train_data_path,
    #                                               test_data_type=args.test_data_type,
    #                                               test_data_path=args.test_data_path,
    #                                               num_of_workers=args.num_of_workers,
    #                                               seed=args.seed)
    # data_loader = datasets_generator.getDataLoader()
    # model = config.model().to(device)
    # logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    # optimizer = config.optimizer(model.parameters())
    # scheduler = config.scheduler(optimizer)
    # criterion = config.criterion()
    # if args.perturb_type == 'samplewise':
    #     train_target = 'train_dataset'
    # else:
    #     if args.use_subset:
    #         data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
    #                                                                train_shuffle=True, train_drop_last=True)
    #         train_target = 'train_subset'
    #     else:
    #         data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
    #         train_target = 'train_dataset'

    # trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    # evaluator = Evaluator(data_loader, logger, config)

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    # if args.data_parallel:
    #     model = torch.nn.DataParallel(model)

    # if args.load_model:
    #     checkpoint = util.load_model(filename=checkpoint_path_file,
    #                                  model=model,
    #                                  optimizer=optimizer,
    #                                  alpha_optimizer=None,
    #                                  scheduler=scheduler)
    #     ENV = checkpoint['ENV']
    #     trainer.global_step = ENV['global_step']
    #     logger.info("File %s loaded!" % (checkpoint_path_file))

    # data prepare
    # random_noise_class = np.load('noise_class_label.npy')
    # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    # # we have to change the target randomly to give the noise a label
    # train_data.replace_random_noise_class(random_noise_class)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    # train_noise_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
    # train_noise_data.replace_random_noise_class(random_noise_class)
    # train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test data don't have to change the target. by renjie3
    # memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
    # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    # model setup and optimizer config
    model = Model(feature_dim, arch=args.arch).cuda()
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # load pre-trained model parameters here by renjie3.
    # unlearnable_20211011011237_0.5_512_150
    # unlearnable_36176425_20211102011903_0.5_512_1000_statistics
    pre_load_name = "unlearnable_samplewise_104763711_20211115145540_0.5_512_1000"
    pretrained_model_path = "./results/{}_model.pth".format(pre_load_name)
    model.load_state_dict(torch.load(pretrained_model_path))
    perturbation_budget = 16
    # # load noise here:
    pretrained_samplewise_noise = torch.load("./results/unlearnable_samplewise_104763711_20211115145540_0.5_512_1000perturbation.pt")
    # random_noise_class_path = 'noise_class_label_test.npy'

    # train_data = utils.TransferCIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, perturb_tensor_filepath="./results/{}_checkpoint_perturbation.pt".format(pre_load_name), random_noise_class_path=random_noise_class_path, perturbation_budget=perturbation_budget, class_4=False)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # c = len(memory_data.classes)

    # training loop
    # results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    # save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    # if not os.path.exists('results'):
    #     os.mkdir('results')
    # best_acc = 0.0
    # for epoch in range(1, epochs + 1):
        # train_loss = train_simclr(model, train_loader, optimizer)
        # results['train_loss'].append(train_loss)
        # test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader)
        # results['test_acc@1'].append(test_acc_1)
        # results['test_acc@5'].append(test_acc_5)
        # save statistics
        # data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        # data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        # if test_acc_1 > best_acc:
        #     best_acc = test_acc_1
        #     torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        if args.attack_type == 'min-max':
            # min-max noise need model to converge first. ssl don't need this yes 20210926
            train(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)
        if args.random_start:
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        else:
            random_noise = torch.zeros(*args.noise_shape)
        if args.perturb_type == 'samplewise':
            # noise = sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV)
            pass
        elif args.perturb_type == 'classwise':
            # noise, save_name_pre = universal_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k)
            # save_name_pre = ''
            # create new test dataset
            # train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
            test_data_visualization = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True, class_4=True)
            # test_data_visualization_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
            # load model here:
            random_noise_class_test = np.load('noise_class_label_test.npy')
            # load noise here:
            # pretrained_classwise_noise = None
            # test_visualization
            # test_ssl_visualization(model, test_data_visualization, random_noise_class_test, pretrained_classwise_noise, pre_load_name+"retrain", True)
            plot_distribution(model, test_data_visualization, pretrained_samplewise_noise, pre_load_name+"_feature")
            # test_ssl_visualization(model, train_data, None, None, pre_load_name)

        # torch.save(noise, os.path.join(args.exp_name, save_name_pre+'_perturbation.pt'))
        
        # torch.save(net.state_dict(), args.model_parameters_path)
        # logger.info(noise)
        # logger.info(noise.shape)
        # logger.info('Noise saved at %s' % (os.path.join(args.exp_name, save_name_pre+'_perturbation.pt')))
    else:
        raise('Not implemented yet')
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
