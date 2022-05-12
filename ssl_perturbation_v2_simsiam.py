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
parser.add_argument('--perturb_type', default='classwise', type=str, help='Perturb type')
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
parser.add_argument('--epochs', default=300, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
# parser.add_argument('--noise_num', default=10, type=int, help='The number of categories of misleading noise')
parser.add_argument('--save_image_num', default=10, type=int, help='The number of groups of images with noise to save every 10 epochs. Evrey gourp has 9 images')
parser.add_argument('--min_min_attack_fn', default="non_eot", type=str, help='The function of min_min_attack')
parser.add_argument('--strong_aug', action='store_true', default=False)
parser.add_argument('--local_dev', default='', type=str, help='The gpu number used on developing node.')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--model_group', default=1, type=int, help='The number of models to be used train unlearnable.')
parser.add_argument('--job_id', default='', type=str, help='The Slurm JOB ID')
parser.add_argument('--org_label_noise', action='store_true', default=False, help='Using original label to allocate noise class')
parser.add_argument('--class_4', action='store_true', default=False)
parser.add_argument('--class_4_train_size', default=1024, type=int, help='The size of training set for 4class')
parser.add_argument('--noise_after_transform', action='store_true', default=False)
parser.add_argument('--shuffle_train_perturb_data', action='store_true', default=False)
parser.add_argument('--not_shuffle_train_data', action='store_true', default=False)
parser.add_argument('--shuffle_step', default=5, type=int, help='Reshuffle the idx every n steps.')
parser.add_argument('--perturb_first', action='store_true', default=False)
parser.add_argument('--num_den_sheduler', default=[0], nargs='+', type=int, help='numerator denomerator alternately')
parser.add_argument('--plot_process', action='store_true', default=False)
parser.add_argument('--plot_process_mode', default='pair', type=str, choices=['pair', 'augmentation', 'center'], help='What samples to plot')
parser.add_argument('--plot_process_feature', default='out', type=str, choices=['feature', 'out'], help='What to plot? Feature or out?')
# parser.add_argument('--mix', default='no', type=str, choices=['no', 'all_mnist', 'train_mnist', 'test_mnist', 'train_mnist_10_128', 'all_mnist_10_128', 'all_mnist_18_128', 'train_mnist_18_128', 'samplewise_all_mnist_18_128', 'samplewise_train_mnist_18_128', 'concat_samplewise_train_mnist_18_128', 'concat_samplewise_all_mnist_18_128', 'concat4_samplewise_train_mnist_18_128', 'concat4_samplewise_all_mnist_18_128', 'mnist', 'samplewise_all_center_8_64', 'samplewise_train_center_8_64',   'samplewise_all_corner_8_64', 'samplewise_train_corner_8_64',  'samplewise_all_center_10_128', 'samplewise_train_center_10_128', 'samplewise_all_corner_10_128', 'samplewise_train_corner_10_128', 'all_center_10_128', 'train_center_10_128', 'all_corner_10_128', 'train_corner_10_128'], help='Add new features to data')
parser.add_argument('--mix', default='no', type=str, help='Add new features to data')
parser.add_argument('--load_model_path', default='', type=str, help='load_model_path')
parser.add_argument('--load_model_path2', default='', type=str, help='load_model_path')
parser.add_argument('--just_test', action='store_true', default=False)
parser.add_argument('--plot_beginning_and_end', action='store_true', default=False)
parser.add_argument('--plot_be_mode', default='ave_augmentation', type=str, help='What samples to plot')
parser.add_argument('--plot_be_mode_feature', default='out', type=str, choices=['feature', 'out'], help='What to plot? Feature or out?')
parser.add_argument('--gray_train', default='no', type=str, help='gray_train')
# parser.add_argument('--gray_test', default='no', type=str, choices=['gray', 'no', 'red', 'gray_mnist', 'grayshift_mnist', 'colorshift_mnist', 'grayshift_font_mnist', 'grayshift2_font_mnist', 'grayshift_font_singledigit_mnist', 'grayshift_font_randomdigit_mnist', 'grayshiftlarge_font_randomdigit_mnist', 'grayshiftlarge_font_singldigit_mnist'], help='gray_test')
parser.add_argument('--gray_test', default='no', type=str, help='gray_test')
parser.add_argument('--augmentation', default='simclr', type=str, help='What')
parser.add_argument('--augmentation_prob', default=[0, 0, 0, 0], nargs='+', type=float, help='get augmentation by probility')
parser.add_argument('--n_zspace', default=3, type=int, help='the number of Z spaces')
parser.add_argument('--batchsize_2digit', default=256, type=int, help='batchsize_2digit')
parser.add_argument('--theory_normalize', action='store_true', default=False)
parser.add_argument('--theory_train_data', default='hierarchical_knn4', type=str, help='What theory data to use')
parser.add_argument('--theory_test_data', default='hierarchical_test_knn4', type=str, help='What theory data to use')
parser.add_argument('--random_drop_feature_num', default=[0, 0, 0, 0, 0], nargs='+', type=int, help='the number of randomly dropped features')
parser.add_argument('--gaussian_aug_std', default=0.05, type=float, help='Std of 0-mean gaussian augmentation.')
parser.add_argument('--thoery_schedule_dim', default=90, type=int, help='the dimenssion of schedule')
parser.add_argument('--just_test_temp_save_file', default='temp', type=str, help='just_test_temp_save_file')
parser.add_argument('--save_name_pre_temp_save_file', default='temp', type=str, help='save_name_pre_temp_save_file')
parser.add_argument('--theory_aug_by_order', action='store_true', default=False)
parser.add_argument('--just_test_plot', action='store_true', default=False)
parser.add_argument('--save_random', action='store_true', default=False)
parser.add_argument('--cross_eot', action='store_true', default=False)
parser.add_argument('--test_cluster_dim_range', default=[0, 10], nargs='+', type=int, help='the dimenssion range of test data')
parser.add_argument('--eot_size', default=30, type=int, help='the dimenssion range of test data')
parser.add_argument('--one_gpu_eot_times', default=1, type=int, help='the dimenssion range of test data')
parser.add_argument('--split_transform', action='store_true', default=False)
parser.add_argument('--pytorch_aug', action='store_true', default=False)
parser.add_argument('--dbindex_weight', default=0, type=float, help='dbindex_weight')
parser.add_argument('--kmeans_index', default=-1, type=int, help='whether to use kmeans label and which group to use')
parser.add_argument('--kmeans_index2', default=-1, type=int, help='whether to use kmeans label and which group to use')
parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
parser.add_argument('--cluster_wise', action='store_true', default=False)
parser.add_argument('--pre_load_noise_name', default='', type=str, help='Usually used with a pretrained model')
parser.add_argument('--n_cluster', default=20, type=int, help='num_workers')
parser.add_argument('--dbindex_label_index', default=2, type=int, help='num_workers')
parser.add_argument('--noise_dbindex_weight', default=0, type=float, help='dbindex_weight')
parser.add_argument('--simclr_weight', default=1, type=float, help='dbindex_weight')
parser.add_argument('--clean_weight', default=0, type=float, help='dbindex_weight')
parser.add_argument('--noise_after_transform_dataset', action='store_true', default=False)
parser.add_argument('--noise_after_transform_noise_dataset', action='store_true', default=False)
parser.add_argument('--noise_simclr_weight', default=0, type=float, help='noise_simclr_weight')
parser.add_argument('--noise_after_transform_train_model', action='store_true', default=False)
parser.add_argument('--double_perturb', action='store_true', default=False)
parser.add_argument('--upper_half_linear', action='store_true', default=False)
parser.add_argument('--linear_style', default='upper_half_linear', type=str, help='num_workers')
parser.add_argument('--mask_linear_constraint', action='store_true', default=False)
parser.add_argument('--mask_linear_noise_range', default=[2, 8], nargs='+', type=float, help='the dimenssion range of test data')
parser.add_argument('--use_supervised_g', action='store_true', default=False)
parser.add_argument('--supervised_weight', default=1, type=float, help='noise_simclr_weight')
parser.add_argument('--linear_model_g', action='store_true', default=False)
parser.add_argument('--linear_noise_dbindex_weight', default=0, type=float, help='noise_simclr_weight')
parser.add_argument('--linear_noise_dbindex_index', default=1, type=int, help='noise_simclr_weight')
parser.add_argument('--linear_noise_dbindex_weight2', default=0, type=float, help='noise_simclr_weight')
parser.add_argument('--linear_noise_dbindex_index2', default=2, type=int, help='noise_simclr_weight')
parser.add_argument('--save_kmeans_label', action='store_true', default=False)
parser.add_argument('--unlearnable_kmeans_label', action='store_true', default=False)
parser.add_argument('--kmeans_label_file', default='', type=str, help='just_test_temp_save_file')
parser.add_argument('--skip_train_model', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--not_use_mean_dbindex', action='store_true', default=False)
parser.add_argument('--not_use_normalized', action='store_true', default=False)
parser.add_argument('--use_wholeset_center', action='store_true', default=False)
parser.add_argument('--modify_dbindex', default='', type=str, help='just_test_temp_save_file')
parser.add_argument('--two_stage_PGD', action='store_true', default=False)
parser.add_argument('--model_g_augment_first', action='store_true', default=False)
parser.add_argument('--dbindex_augmentation', action='store_true', default=False)
parser.add_argument('--linear_xnoise_dbindex_weight', default=0, type=float, help='noise_simclr_weight')
parser.add_argument('--linear_xnoise_dbindex_index', default=1, type=int, help='noise_simclr_weight')

parser.add_argument('--no_eval', action='store_true', default=False)

parser.add_argument('--single_noise_after_transform', action='store_true', default=False)

parser.add_argument('--load_piermaro_model', action='store_true', default=False)
parser.add_argument('--load_piermaro_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--piermaro_whole_epoch', default='', type=str, help='Whole epoch when use re_job to train')
parser.add_argument('--piermaro_restart_epoch', default=0, type=int, help='The order of epoch when use re_job to train')

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
from utils import train_diff_transform
import datetime
from model import Model, LooC, TheoryModel, MICL, LinearModel
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import test_ssl, train_simclr, train_simclr_noise_return_loss_tensor, train_simclr_target_task, train_simclr_softmax, test_ssl_softmax, train_align, train_looc, train_micl, train_simclr_newneg, train_simclr_2digit, test_intra_inter_sim, test_instance_sim, train_simclr_theory, test_ssl_theory, test_instance_sim_thoery, test_cluster, find_cluster
import random
import matplotlib.pyplot as plt
import matplotlib
from thop import profile, clever_format

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils import train_supervised_batch
from supervised_models import *
from torchvision import transforms
from simsiam_utils import train_simsiam, test_simsiam
from methods import set_model

import pickle

mlconfig.register(madrys.MadrysLoss)

# Convert Eps
args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255
flag_shuffle_train_data = not args.not_shuffle_train_data
flag_use_normalized = not args.not_use_normalized
flag_use_mean_dbindex = not args.not_use_mean_dbindex

# Set up Experiments
if args.load_model_path == '' and args.load_model:
    # args.exp_name = 'exp_' + datetime.datetime.now()
    raise('Use load file name!')
if args.plot_beginning_and_end and not args.load_model:
    raise('Load pretrained model!')

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
if not args.no_save:
    logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    if not args.no_save:
        logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
if not args.no_save:
    for key in config:
        logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


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


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, const_train_loader, save_name_pre):
    # Class-Wise perturbation
    # Generate Data loader

    condition = True
    # data_iter = iter(data_loader['train_dataset'])
    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    if save_name_pre == None:
        if args.job_id == '':
            save_name_pre = 'unlearnable_classwise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
        else:
            save_name_pre = 'unlearnable_classwise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)

    if args.load_piermaro_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
        best_loss = results['best_loss'][len(results['best_loss'])-1]
        best_loss_acc = results['best_loss_acc'][len(results['best_loss_acc'])-1]
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': [], "numerator": [], "denominator": []}
    
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    cluster_label_index = args.dbindex_label_index
    for _epoch_idx in range(1, epochs+1):
        epoch_idx = _epoch_idx + args.piermaro_restart_epoch

        flag_cluster = False
        if args.cluster_wise:
            if args.load_model and epoch_idx == 1:
                if args.dbindex_label_index == 2:
                    kmeans_labels = find_cluster(model, const_train_loader, random_noise, args.n_cluster, label_index=0)
                    train_noise_data_loader_simclr.dataset.add_kmeans_label(kmeans_labels)
                    train_loader_simclr.dataset.add_kmeans_label(kmeans_labels)
                    flag_cluster = True
                    classwise_random_noise = []
                    for _i in range(args.n_cluster):
                        idx = np.where(kmeans_labels == _i)[0]
                        classwise_random_noise.append(random_noise[idx].mean(dim=0))
                    random_noise = torch.stack(classwise_random_noise, dim=0)
                else:
                    classwise_random_noise = []
                    for _i in range(10):
                        print(_i)
                        idx = np.where(const_train_loader.dataset.targets == _i)[0]
                        classwise_random_noise.append(random_noise[idx].mean(dim=0))
                    random_noise = torch.stack(classwise_random_noise, dim=0)

        # print(epoch_idx, condition)
        # for item in train_loader_simclr:
            # print("item shape: ", item[0].shape)
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0,0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0
        # logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
        condition = True
        if hasattr(model, 'classify'):
            model.classify = True

        while condition:
            if args.attack_type == 'min-min':
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
                        class_noise = random_noise[label[cluster_label_index].item()].to(device)
                        # mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_1.shape, patch_location=args.patch_location)
                        train_pos_1.append(pos_samples_1[i]+class_noise)
                        train_pos_2.append(pos_samples_2[i]+class_noise)
                    # Train
                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    # trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)
                    batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count
                    sum_numerator += numerator
                    sum_numerator_count += 1
                    sum_denominator += denominator
                    sum_denominator_count += 1
                
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training perturbation"):
            # for i, (pos_samples_1, pos_samples_2, labels) in enumerate(train_noise_data_loader_simclr):
                # print(i, "one noise batch")
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)
                # Add Class-wise Noise to each sample
                batch_noise, mask_cord_list = [], []
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    class_noise = random_noise[label[cluster_label_index].item()]
                    # mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_1.shape, patch_location=args.patch_location)
                    batch_noise.append(class_noise)
                    # mask_cord_list.append(mask_cord)

                # Update universal perturbation
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                batch_noise = torch.stack(batch_noise).to(device)
                if flag_cluster:
                    dbindex_weight = args.dbindex_weight
                else:
                    dbindex_weight = 0

                if args.attack_type == 'min-min':
                    if args.min_min_attack_fn == "eot_v1":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simsiam_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform, eot_size=args.eot_size, one_gpu_eot_times=args.one_gpu_eot_times, cross_eot=args.cross_eot, pytorch_aug=args.pytorch_aug, dbindex_weight=dbindex_weight, single_noise_after_transform=args.single_noise_after_transform, no_eval=args.no_eval, dbindex_label_index=args.dbindex_label_index, noise_dbindex_weight=args.noise_dbindex_weight)
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform)
                    elif args.min_min_attack_fn in ["pos/neg", "pos", "neg"]:
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, target_task=args.min_min_attack_fn)
                else:
                    raise('Invalid attack')

                # print("eta: {}".format(np.mean(np.absolute(eta.mean(dim=0).to('cpu').numpy())) * 255))
                class_noise_eta = collections.defaultdict(list)
                for i in range(len(eta)):
                    # x1, x2, y1, y2 = mask_cord_list[i]
                    delta = eta[i]#[:, x1: x2, y1: y2]
                    class_noise_eta[labels[i, cluster_label_index].item()].append(delta.detach().cpu())

                random_noise = random_noise.to('cpu')
                for key in class_noise_eta:
                    delta = torch.stack(class_noise_eta[key]).mean(dim=0) - random_noise[key]#. For delta, we didn't use absolute before mean
                    # print("org_class_noise_eta[key]: {}".format(np.mean(np.absolute(torch.stack(class_noise_eta[key]).to('cpu').numpy())) * 255))
                    # print("org_random_noise[key]: {}".format(np.mean(np.absolute(random_noise[key].to('cpu').numpy())) * 255))
                    class_noise = random_noise[key]
                    class_noise += delta
                    # print("check delta before clamp: {}".format(np.mean(np.absolute(delta.to('cpu').numpy())) * 255))
                    # print("check random_noise before clamp: {}".format(np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255))
                    random_noise[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon) # important.
                    # print("check random_noise: {}".format(np.mean(np.absolute(random_noise[key].to('cpu').numpy())) * 255))
                # print("check all random_noise after clamp: {}".format(np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255))
                noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255
                
                # input()
            # print(train_noise_loss_sum / float(train_noise_loss_count))
            
        # # Here we save some samples in image.
        # if epoch_idx % 10 == 0 and not args.no_save:
        # # if True:
        #     if not os.path.exists('./images/'+save_name_pre):
        #         os.mkdir('./images/'+save_name_pre)
        #     images = []
        #     for group_idx in range(save_image_num):
        #         utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))

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
        numerator = sum_numerator / float(sum_numerator_count)
        denominator = sum_denominator / float(sum_denominator_count)
        print("train_loss:", train_loss)
        results['train_loss'].append(train_loss)
        results['noise_ave_value'].append(noise_ave_value)
        test_acc_1, test_acc_5 = test_simsiam(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['best_loss'].append(best_loss)
        results['best_loss_acc'].append(best_loss_acc)

        results['numerator'].append(numerator)
        results['denominator'].append(denominator)

        # print("results['numerator']", results['numerator'])
        # print("results['denominator']", results['denominator'])

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 1 == 0 and not args.no_save:
            torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            print("model saved at " + save_name_pre)

        if epoch_idx % 10 == 0 and not args.no_save:
            torch.save(model.state_dict(), 'results/{}_checkpoint_model_epoch_{}.pth'.format(save_name_pre, epoch_idx))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation_epoch_{}.pt'.format(save_name_pre, epoch_idx))
            print("model saved at " + save_name_pre)
    
    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

        piermaro_checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'perturbation': random_noise}
        torch.save(piermaro_checkpoint, 'results/{}_piermaro_model.pth'.format(save_name_pre))

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


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, save_name_pre, const_train_loader, g_net,  supervised_criterion, supervised_optimizer, supervised_scheduler, supervised_transform_train):
    # mask_cord_list = []
    # idx = 0
    # for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
    #     for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
    #         if args.shuffle_train_perturb_data:
    #             noise = random_noise[label.item()]
    #         else:
    #             noise = random_noise[idx]
    #         mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
    #         mask_cord_list.append(mask_cord)
    #         idx += 1

    if args.upper_half_linear:
        linear_separable_noise, simclr_mask = utils.linear_separable_perturbation([32, 32], [4,4], 10, 5000, const_train_loader.dataset, linear_style=args.linear_style)
        linear_separable_noise, simclr_mask = linear_separable_noise.to(device), simclr_mask.to(device)
    else:
        linear_separable_noise = None
        simclr_mask = None

    if args.mask_linear_constraint:
        mask1, mask2 = utils.get_linear_constraint_mask([32, 32], [4,4], 10, 5000, const_train_loader.dataset,)
 
    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    if save_name_pre == None:
        if args.job_id == '':
            save_name_pre = 'unlearnable_samplewise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
        else:
            save_name_pre = 'unlearnable_samplewise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)

    if args.load_piermaro_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
        best_loss = results['best_loss'][len(results['best_loss'])-1]
        best_loss_acc = results['best_loss_acc'][len(results['best_loss_acc'])-1]
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': [], "numerator": [], "denominator": []}
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    flag_cluster = False

    test_acc_1, test_acc_5 = test_simsiam(model, memory_loader, test_loader, k, temperature, 0, epochs)

    for _epoch_idx in range(1, epochs+1):
        epoch_idx = _epoch_idx + args.piermaro_restart_epoch
        train_idx = 0
        condition = True
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0, 0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0

        # flag_cluster = False
        if args.cluster_wise:
            if (args.load_model and epoch_idx == 1) or (not args.load_model and epoch_idx == 3):
                kmeans_labels = find_cluster(model, const_train_loader, random_noise, args.n_cluster)
                train_noise_data_loader_simclr.dataset.add_kmeans_label(kmeans_labels)
                flag_cluster = True
        if args.save_kmeans_label:
            kmeans_labels1 = find_cluster(model, const_train_loader, random_noise, 10)
            kmeans_labels2 = find_cluster(model, const_train_loader, random_noise, 100)
            kmeans_labels3 = find_cluster(model, const_train_loader, random_noise, 500)
            kmeans_labels = np.stack([kmeans_labels1, kmeans_labels2, kmeans_labels3], axis=0)
            # f = open('./data/kmeans_label/kmeans_unlearnable_simclr_label.pkl', 'wb')
            # pickle.dump(kmeans_labels, f)
            # f.close()
            input('kmeans_unlearnable_simclr_label done')

        while condition:
            if args.attack_type == 'min-min':
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    _start = time.time()
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

                    if args.skip_train_model:
                        continue

                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    if args.noise_after_transform_train_model:
                        pos_samples_1 = utils.train_diff_transform(pos_samples_1)
                        pos_samples_2 = utils.train_diff_transform(pos_samples_2)

                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        sample_noise = random_noise[label[0].item()]
                        # c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                        # mask = np.zeros((c, h, w), np.float32)
                        # x1, x2, y1, y2 = mask_cord_list[train_idx]
                        if type(sample_noise) is np.ndarray:
                            mask = sample_noise
                        else:
                            mask = sample_noise.cpu().numpy()
                        # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        if args.upper_half_linear:
                            sample_noise = torch.from_numpy(mask).to(device) * simclr_mask[label[0].item()] + linear_separable_noise[label[0].item()]
                        else:
                            sample_noise = torch.from_numpy(mask).to(device)
                        
                        # images[i] = images[i] + sample_noise
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)
                        train_idx += 1

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    batch_train_loss, batch_size_count = train_simsiam(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform_train_model, pytorch_aug=False)

                    if args.use_supervised_g:
                        g_net.train()
                        for param in g_net.parameters():
                            param.requires_grad = True
                        train_supervised_batch(g_net, torch.stack(train_pos_1).to(device), labels[:, 1].to(device), supervised_criterion, supervised_optimizer, supervised_transform_train)
                    
                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count
                    # sum_numerator += numerator
                    # sum_numerator_count += 1
                    # sum_denominator += denominator
                    # sum_denominator_count += 1

                    _end = time.time()

                    print("traning model time:", _end - _start)

            # Search For Noise

            if args.use_wholeset_center:
                noise_centroids = utils.get_centers(random_noise, train_noise_data_loader_simclr.dataset.targets[:, args.linear_noise_dbindex_index], flag_use_normalized)
            else:
                noise_centroids = None
            
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            idx = 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)

                if args.debug:
                    print(torch.max(labels[:, args.linear_noise_dbindex_index]))
                    input('check')

                if args.noise_after_transform:
                    # print('check not come noise_after_transform')
                    pos_samples_1 = utils.train_diff_transform(pos_samples_1)
                    pos_samples_2 = utils.train_diff_transform(pos_samples_2)

                if args.single_noise_after_transform:
                    # print('check single aug on 1')
                    pos_samples_1 = utils.train_diff_transform(pos_samples_1)

                # Add Sample-wise Noise to each sample
                batch_noise, batch_start_idx = [], idx
                batch_simclr_mask = []
                batch_linear_noise = []
                batch_mask1 = []
                batch_mask2 = []
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    sample_noise = random_noise[label[0].item()]
                    # c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                    # mask = np.zeros((c, h, w), np.float32)
                    # x1, x2, y1, y2 = mask_cord_list[idx]
                    if type(sample_noise) is np.ndarray:
                        mask = sample_noise
                    else:
                        mask = sample_noise.cpu().numpy()
                    # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    sample_noise = torch.from_numpy(mask).to(device)
                    batch_noise.append(sample_noise)
                    if args.upper_half_linear:
                        batch_simclr_mask.append(simclr_mask[label[0].item()])
                        batch_linear_noise.append(linear_separable_noise[label[0].item()])
                    if args.mask_linear_constraint:
                        batch_mask1.append(mask1[label[1].item()])
                        batch_mask2.append(mask2[label[1].item()])
                    idx += 1

                # Update sample-wise perturbation
                # model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                if args.use_supervised_g:
                    g_net.eval()
                    for param in g_net.parameters():
                        param.requires_grad = False
                batch_noise = torch.stack(batch_noise).to(device)
                if args.upper_half_linear:
                    batch_simclr_mask = torch.stack(batch_simclr_mask).to(device)
                    batch_linear_noise = torch.stack(batch_linear_noise).to(device)
                elif args.mask_linear_constraint:
                    batch_mask1 = torch.stack(batch_mask1).to(device)
                    batch_mask2 = torch.stack(batch_mask2).to(device)
                else:
                    batch_simclr_mask = None
                    batch_linear_noise = None
                if flag_cluster:
                    dbindex_weight = args.dbindex_weight
                else:
                    dbindex_weight = 0
                if args.attack_type == 'min-min':
                    if args.min_min_attack_fn == "eot_v1":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simsiam_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform, eot_size=args.eot_size, one_gpu_eot_times=args.one_gpu_eot_times, cross_eot=args.cross_eot, pytorch_aug=args.pytorch_aug, dbindex_weight=dbindex_weight, single_noise_after_transform=args.single_noise_after_transform, no_eval=args.no_eval, dbindex_label_index=args.dbindex_label_index, noise_dbindex_weight=args.noise_dbindex_weight, simclr_weight=args.simclr_weight, augmentation_prob=args.augmentation_prob, clean_weight=args.clean_weight, noise_simclr_weight=args.noise_simclr_weight, double_perturb=args.double_perturb, upper_half_linear=args.upper_half_linear, batch_simclr_mask=batch_simclr_mask, batch_linear_noise=batch_linear_noise, mask_linear_constraint=args.mask_linear_constraint, mask1=batch_mask1, mask2=batch_mask2, mask_linear_noise_range=args.mask_linear_noise_range, use_supervised_g=args.use_supervised_g, g_net=g_net, supervised_criterion=supervised_criterion, supervised_weight=args.supervised_weight, supervised_transform_train=supervised_transform_train, linear_noise_dbindex_weight=args.linear_noise_dbindex_weight, linear_noise_dbindex_index=args.linear_noise_dbindex_index, linear_noise_dbindex_weight2=args.linear_noise_dbindex_weight2, linear_noise_dbindex_index2=args.linear_noise_dbindex_index2, use_mean_dbindex=flag_use_mean_dbindex, use_normalized=flag_use_normalized, noise_centroids=noise_centroids, modify_dbindex=args.modify_dbindex, two_stage_PGD=args.two_stage_PGD, model_g_augment_first=args.model_g_augment_first, dbindex_augmentation=args.dbindex_augmentation, linear_xnoise_dbindex_weight=args.linear_xnoise_dbindex_weight, linear_xnoise_dbindex_index=args.linear_xnoise_dbindex_index,)
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform, split_transform=args.split_transform)
                    else:
                        raise('Using wrong min_min_attack_fn in samplewise.')
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                for delta, label in zip(eta, labels):
                    # x1, x2, y1, y2 = mask_cord_list[labels[i].item()]
                    # delta = delta[:, x1: x2, y1: y2]
                    if torch.is_tensor(random_noise):
                        random_noise[label[0].item()] = delta.detach().cpu().clone()
                    else:
                        random_noise[label[0].item()] = delta.detach().cpu().numpy()
                    # print(np.sum(np.isnan(delta.detach().cpu().numpy())))
                    # print(delta.detach().cpu().numpy())

                noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255

            # if args.debug:
            #     test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, 0, epochs)
                # print("noise_ave_value", noise_ave_value)
        
        train_loss = sum_train_loss / float(sum_train_batch_size)
        # numerator = sum_numerator / float(sum_numerator_count)
        # denominator = sum_denominator / float(sum_denominator_count)
        print(train_loss)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_simsiam(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['noise_ave_value'].append(noise_ave_value)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['best_loss'].append(best_loss)
        results['best_loss_acc'].append(best_loss_acc)

        results['numerator'].append(0)
        results['denominator'].append(0)

        # print("results['numerator']", results['numerator'])
        # print("results['denominator']", results['denominator'])

        # save statistics
        # print(results)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 1 == 0 and not args.no_save:
            torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            if args.upper_half_linear:
                whole_noise = random_noise * simclr_mask.detach().cpu() + linear_separable_noise.detach().cpu()
                torch.save(whole_noise, 'results/{}_checkpoint_{}_perturbation.pt'.format(save_name_pre, args.linear_style))
            print("model saved at " + save_name_pre)

        if epoch_idx % 10 == 0 and not args.no_save:
            torch.save(model.state_dict(), 'results/{}_checkpoint_model_epoch_{}.pth'.format(save_name_pre, epoch_idx))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation_epoch_{}.pt'.format(save_name_pre, epoch_idx))
            if args.upper_half_linear:
                whole_noise = random_noise * simclr_mask.detach().cpu() + linear_separable_noise.detach().cpu()
                torch.save(whole_noise, 'results/{}_checkpoint_{}_perturbation.pt'.format(save_name_pre, args.linear_style))
            print("model saved at " + save_name_pre)

        if args.use_supervised_g:
            supervised_scheduler.step()

    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

        piermaro_checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'perturbation': random_noise}
        torch.save(piermaro_checkpoint, 'results/{}_piermaro_model.pth'.format(save_name_pre))

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            # c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
            # mask = np.zeros((c, h, w), np.float32)
            # x1, x2, y1, y2 = mask_cord_list[idx]
            mask = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise, save_name_pre
    else:
        return random_noise, save_name_pre

def just_test(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, plot_input_data_loader):

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_justtest_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_justtest_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
        
    intra_sim, inter_sim = test_intra_inter_sim(model, memory_loader, train_loader_simclr, k, temperature, epochs, distance=False)
    intra_dis, inter_dis = test_intra_inter_sim(model, memory_loader, train_loader_simclr, k, temperature, epochs, distance=True)
    print(intra_sim / inter_sim)
    print(intra_dis / inter_dis)
    acc_top1, acc_top5 = test_instance_sim(model, memory_loader, train_loader_simclr, k, temperature, epochs, augmentation=args.augmentation, augmentation_prob=args.augmentation_prob)
    print(acc_top1, acc_top5)

    return random_noise, save_name_pre

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

    if args.use_supervised_g:
        # Data
        print('==> Preparing data..')
        supervised_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
        ])

        if args.train_data_type == 'CIFAR10':
            if args.class_4:
                args.num_class = 4
            else:
                args.num_class = 10
        elif args.train_data_type == 'CIFAR100':
            args.num_class = 100
        else:
            raise('Wrong train_data_type')

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        print('==> Building model..')

        if args.linear_model_g:
            g_net = LinearModel([3, 32, 32], args.num_class)
        else:
            g_net = ResNet18(args.num_class)
        
        g_net = g_net.to(device)

        supervised_criterion = nn.CrossEntropyLoss()
        supervised_optimizer = optim.SGD(g_net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4)
        supervised_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(supervised_optimizer, T_max=epochs)
    else:
        supervised_transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
        ])
        g_net = None
        supervised_criterion = None
        supervised_optimizer = None
        supervised_scheduler = None

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    
    if args.class_4:
        random_noise_class = np.load('noise_class_label_1024_4class.npy')
    else:
        random_noise_class = np.load('noise_class_label.npy')

    if args.train_data_type == 'CIFAR10':
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform_dataset, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size, kmeans_index=args.kmeans_index, kmeans_index2=args.kmeans_index2, unlearnable_kmeans_label=args.unlearnable_kmeans_label, kmeans_label_file=args.kmeans_label_file)
        train_data.replace_targets_with_id()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        const_train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=False, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size, kmeans_index=args.kmeans_index, kmeans_index2=args.kmeans_index2, unlearnable_kmeans_label=args.unlearnable_kmeans_label, kmeans_label_file=args.kmeans_label_file)
        const_train_data.replace_targets_with_id()
        const_train_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        train_noise_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform_noise_dataset, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size, kmeans_index=args.kmeans_index, kmeans_index2=args.kmeans_index2, unlearnable_kmeans_label=args.unlearnable_kmeans_label, kmeans_label_file=args.kmeans_label_file)
        
        train_noise_data.replace_targets_with_id()
        
        train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=args.num_workers, pin_memory=True)
        # test data don't have to change the target. by renjie3
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test, train_noise_after_transform=False, )
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test, train_noise_after_transform=False, )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        plot_input_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=False, gray=args.gray_test, class_4_train_size=args.class_4_train_size)
        plot_input_data_loader = DataLoader(plot_input_data, batch_size=1024, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    elif args.train_data_type == 'CIFAR100':
        train_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=args.noise_after_transform_dataset, kmeans_index=args.kmeans_index, unlearnable_kmeans_label=args.unlearnable_kmeans_label)
        train_data.replace_targets_with_id()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        const_train_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=False, kmeans_index=args.kmeans_index, unlearnable_kmeans_label=args.unlearnable_kmeans_label)
        const_train_data.replace_targets_with_id()
        const_train_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        train_noise_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=args.noise_after_transform_noise_dataset, kmeans_index=args.kmeans_index, unlearnable_kmeans_label=args.unlearnable_kmeans_label)
        
        train_noise_data.replace_targets_with_id()
        
        train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=args.num_workers, pin_memory=True)
        # test data don't have to change the target. by renjie3
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=False, )
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=False, )
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        plot_input_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, train_noise_after_transform=False)
        plot_input_data_loader = DataLoader(plot_input_data, batch_size=1024, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                                num_steps=args.num_steps,
                                                step_size=args.step_size)

    # model = Model(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size)
    # model = model.cuda()
    model = set_model('resnet18', 'cifar10')
    model = model.cuda()

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))

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

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    if args.load_model or args.load_piermaro_model:
        if 'optimizer' in checkpoints:
            optimizer.load_state_dict(checkpoints['optimizer'])

    # flops, params = profile(model, inputs=(torch.randn(6, 1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    c = len(memory_data.classes)

    if args.piermaro_whole_epoch != '':
        if args.load_piermaro_model:
            save_name_pre = args.load_piermaro_model_path
            save_name_pre = save_name_pre.replace("_piermaro_model", "").replace("_model", "")
        else:
            save_name_pre = None
    else:
        save_name_pre = None
    
    # if args.load_model:
    #     # unlearnable_cleantrain_41501264_1_20211204151414_0.5_512_1000_final_model
    #     load_model_path = './results/{}.pth'.format(args.load_model_path)
    #     checkpoints = torch.load(load_model_path, map_location=device)
    #     model.load_state_dict(checkpoints)
    #     # input(load_model_path)
    #     # ENV = checkpoint['ENV']
    #     # trainer.global_step = ENV['global_step']
    #     logger.info("File %s loaded!" % (load_model_path))

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
        
    if args.plot_beginning_and_end:
        plot_be(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
    else:
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
                random_noise = noise_generator.random_noise(noise_shape=args.noise_shape).to(torch.device('cpu'))
                # print(random_noise.device)
            else:
                random_noise = torch.zeros(*args.noise_shape)

            if args.pre_load_noise_name != '':
                random_noise = torch.load("./results/{}.pt".format(args.pre_load_noise_name))

            if args.load_piermaro_model:
                if 'perturbation' in checkpoints:
                    random_noise = checkpoints['perturbation']
                    print('piermaro random_noise loaded')
            
            if args.perturb_type == 'samplewise':
                noise, save_name_pre = sample_wise_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, save_name_pre, const_train_loader, g_net,  supervised_criterion, supervised_optimizer, supervised_scheduler, supervised_transform_train)

            elif args.perturb_type == 'samplewise_dbindex':
                noise, save_name_pre = sample_wise_perturbation_dbindex(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)

            elif args.perturb_type == 'samplewise_mix_stage':
                noise, save_name_pre = sample_wise_perturbation_mix_stage(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)
                
            elif args.perturb_type == 'samplewise_model_free':
                noise, save_name_pre = sample_wise_model_free_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)

            elif args.perturb_type == 'samplewise_myshuffle':
                noise, save_name_pre = sample_wise_perturbation_myshuffle(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, train_data)

            elif args.perturb_type == 'clean_train':
                noise, save_name_pre = clean_train(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'clean_train_newneg':
                noise, save_name_pre = clean_train_newneg(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
            
            elif args.perturb_type == 'clean_train_2digit':
                noise, save_name_pre = clean_train_2digit(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader, args.batchsize_2digit)
                
            elif args.perturb_type == 'theory_model': 
                noise, save_name_pre = theory_model(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
            
            elif args.perturb_type == 'multiple_independent_cl': 
                noise, save_name_pre = micl(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'clean_train_softmax':
                noise, save_name_pre = clean_train_softmax(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'just_test':
                just_test(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'plot_be':
                noise, save_name_pre = plot_be(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'test_find_positive_pair':
                top1_acc, top5_acc, pos_sim, neg_sim = test_find_positive_pair(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                print("The test top1 acc is {}. \nThe test top5 acc is {}. \nAverage postive cosine similarity is {}. \nAverage negative cosine similarity is {}.".format(top1_acc, top5_acc, pos_sim, neg_sim))
                
            elif args.perturb_type == 'looc':
                noise, save_name_pre = looc(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'classwise':
                # noise = universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV)
                if args.model_group > 1:
                    noise, save_name_pre = universal_perturbation_model_group(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)
                else:
                    noise, save_name_pre = universal_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, const_train_loader, save_name_pre)

            else:
                raise('wrong perturb_type')
            if not args.no_save:
                torch.save(noise, 'results/{}perturbation.pt'.format(save_name_pre))
                # logger.info(noise)
                logger.info(noise.shape)
                logger.info('Noise saved at %s' % 'results/{}perturbation.pt'.format(save_name_pre))
            f = open("results_save_name_pre/{}.txt".format(args.job_id), "a")
            f.write("{}".format(save_name_pre))
            f.close()
        else:
            raise('Not implemented yet')
    return


if __name__ == '__main__':
    # for arg in vars(args):
    #     logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
