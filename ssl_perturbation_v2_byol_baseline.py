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
parser.add_argument('--cl_algorithm', default='simclr', type=str, help='just_test_temp_save_file')

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
from byol_pytorch import BYOL
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import test_ssl, train_simclr, train_simclr_noise_return_loss_tensor, train_simclr_target_task, train_simclr_softmax, test_ssl_softmax, train_align, train_looc, train_micl, train_simclr_newneg, train_simclr_2digit, test_intra_inter_sim, test_instance_sim, train_simclr_theory, test_ssl_theory, test_instance_sim_thoery, test_cluster, find_cluster
from byol_utils import train_byol, test_byol
import random
import matplotlib.pyplot as plt
import matplotlib
from thop import profile, clever_format

from resnet_cifar import resnet18_test

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

from utils import train_supervised_batch
from supervised_models import *
from torchvision import transforms
import torchvision

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

def sample_unlabelled_images():
    return torch.randn(20, 3, 32, 32).cuda()

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
            save_name_pre = 'unlearnable_byol_samplewise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
        else:
            save_name_pre = 'unlearnable_byol_samplewise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)

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

    test_acc_1, test_acc_5 = test_byol(model, memory_loader, test_loader, k, temperature, 0, epochs)

    for _epoch_idx in range(1, epochs+1):
        epoch_idx = _epoch_idx + args.piermaro_restart_epoch
        train_idx = 0
        condition = True
        # data_iter = tqdm(train_loader_simclr)
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

        # while condition:
            # if args.attack_type == 'min-min':
            #     # Train Batch for min-min noise
            #     end_of_iteration = "END_OF_ITERATION"
            #     for j in range(0, args.train_step):
            #         _start = time.time()
            #         try:
            #             next_item = next(data_iter, end_of_iteration)
            #             if next_item != end_of_iteration:
            #                 (pos_samples_1, pos_samples_2, labels) = next_item
                            
            #             else:
            #                 condition = False
            #                 del data_iter
            #                 break
            #         except:
            #             # data_iter = iter(data_loader['train_dataset'])
            #             # (pos_1, pos_2, labels) = next(data_iter)
            #             raise('train loader iteration problem')

            #         if args.skip_train_model:
            #             continue

        for pos_samples_1, pos_samples_2, labels in tqdm(train_loader_simclr, total=len(train_loader_simclr)):
            pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
            if args.noise_after_transform_train_model:
                pos_samples_1 = utils.train_diff_transform(pos_samples_1)
                pos_samples_2 = utils.train_diff_transform(pos_samples_2)

            model.train()
            batch_train_loss, batch_size_count = train_byol(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform_train_model, pytorch_aug=False)
            
            sum_train_loss += batch_train_loss
            sum_train_batch_size += batch_size_count

        # _end = time.time()

                    # print("traning model time:", _end - _start)
        
        train_loss = sum_train_loss / float(sum_train_batch_size)
        # numerator = 0
        # denominator = 0
        # print(train_loss)
        results['train_loss'].append(0)
        test_acc_1, test_acc_5 = test_byol(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['noise_ave_value'].append(0)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['best_loss'].append(0)
        results['best_loss_acc'].append(0)

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

def main():
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    arch = args.arch

    supervised_transform_train = None
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


    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                                num_steps=args.num_steps,
                                                step_size=args.step_size)

    if args.cl_algorithm == 'simclr':
        model = Model(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size)
        model = model.cuda()
    elif args.cl_algorithm == 'byol':
        
        model = BYOL(resnet18_test(), image_size = 32, hidden_layer = 'avgpool', use_momentum = True)
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

    optimizer = optim.Adam(model.parameters(), lr=3e-4) #, weight_decay=1e-6)

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
