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
parser.add_argument('--local', default='', type=str, help='The gpu number used on developing node.')
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
parser.add_argument('--simclr_weight', default=1, type=float, help='simclr_weight')
parser.add_argument('--kmeans_index', default=-1, type=int, help='whether to use kmeans label and which group to use')
parser.add_argument('--num_workers', default=2, type=int, help='num_workers')
parser.add_argument('--local_rank', type=int, help='local_rank')
parser.add_argument('--gpu_num', type=int, help='gpu_num')
parser.add_argument('--use_dbindex_train_model', action='store_true', default=False)
parser.add_argument('--no_bn', action='store_true', default=False)

parser.add_argument('--load_piermaro_model', action='store_true', default=False)
parser.add_argument('--load_piermaro_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--piermaro_whole_epoch', default='', type=str, help='Whole epoch when use re_job to train')
parser.add_argument('--piermaro_restart_epoch', default=0, type=int, help='The order of epoch when use re_job to train')

args = parser.parse_args()

import collections
import datetime

import os
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

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
from model import Model, LooC, TheoryModel, MICL, ParalellModel
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import test_ssl, train_simclr, train_simclr_noise_return_loss_tensor, train_simclr_target_task, train_simclr_softmax, test_ssl_softmax, train_align, train_looc, train_micl, train_simclr_newneg, train_simclr_2digit, test_intra_inter_sim, test_instance_sim, train_simclr_theory, test_ssl_theory, test_instance_sim_thoery, test_cluster, train_simclr_dbindex
import random
import matplotlib.pyplot as plt
import matplotlib
from thop import profile, clever_format

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel
# from torch.utils.data.distributed import DistributedSampler

# from resnet_big import Model_no_bn

# torch.cuda.set_device(args.local_rank)
# torch.distributed.init_process_group(backend="nccl")

mlconfig.register(madrys.MadrysLoss)

# Convert Eps
args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255
flag_shuffle_train_data = not args.not_shuffle_train_data

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
    torch.cuda.manual_seed(args.seed)
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


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img):
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
    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    for epoch_idx in range(1, epochs+1):
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
                    noise = random_noise[label.item()]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_1.shape, patch_location=args.patch_location)
                    batch_noise.append(class_noise)
                    mask_cord_list.append(mask_cord)

                # Update universal perturbation
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                batch_noise = torch.stack(batch_noise).to(device)
                print("")
                if args.attack_type == 'min-min':
                    if args.min_min_attack_fn == "eot_v1":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                    elif args.min_min_attack_fn == "eot_v2":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v2(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                    elif args.min_min_attack_fn == "eot_v3":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v3(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform)
                    elif args.min_min_attack_fn in ["pos/neg", "pos", "neg"]:
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, target_task=args.min_min_attack_fn)
                    # train_noise_loss_sum += train_noise_loss * pos_samples_1.shape[0]
                    # train_noise_loss_count += pos_samples_1.shape[0]
                    # perturb_img, eta = noise_generator.min_min_attack_simclr(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature)
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                # print("eta: {}".format(np.mean(np.absolute(eta.mean(dim=0).to('cpu').numpy())) * 255))
                class_noise_eta = collections.defaultdict(list)
                for i in range(len(eta)):
                    x1, x2, y1, y2 = mask_cord_list[i]
                    delta = eta[i][:, x1: x2, y1: y2]
                    class_noise_eta[labels[i].item()].append(delta.detach().cpu())

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
            
        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))

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
        test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
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

        if epoch_idx % 10 == 0 and not args.no_save:
            torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            print("model saved at " + save_name_pre)
    
    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

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


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, save_name_pre):
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
    for _epoch_idx in range(1, epochs+1):
        epoch_idx = _epoch_idx + args.piermaro_restart_epoch
        train_idx = 0
        condition = True
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0,0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0
        while condition:
            if args.attack_type == 'min-min' and not args.load_model:
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

                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    if args.noise_after_transform:
                        pos_samples_1 = utils.train_transform_no_totensor(pos_samples_1)
                        pos_samples_2 = utils.train_transform_no_totensor(pos_samples_2)

                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        sample_noise = random_noise[label[0].item()]
                        # c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                        # mask = np.zeros((c, h, w), np.float32)
                        # x1, x2, y1, y2 = mask_cord_list[train_idx]
                        if type(sample_noise) is np.ndarray:
                            sample_noise = torch.from_numpy(sample_noise).to(device)
                        else:
                            sample_noise = sample_noise.cpu().numpy()
                        # # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        sample_noise = torch.from_numpy(sample_noise).to(device)
                        # images[i] = images[i] + sample_noise
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)
                        train_idx += 1

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    if not args.use_dbindex_train_model:
                        batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
                    else:
                        batch_train_loss, batch_size_count, numerator, denominator = train_simclr_dbindex(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform, dbindex_weight=args.dbindex_weight, pytorch_aug=args.pytorch_aug, simclr_weight=args.simclr_weight, labels=labels)

                    # batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
                    # for debug
                    # print("batch_train_loss: ", batch_train_loss / float(batch_size_count))
                    # debug_loss = train_simclr_noise_return_loss_tensor(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature)
                    # print("debug_loss: ", debug_loss.item())
                    # input()
                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count
                    sum_numerator += numerator
                    sum_numerator_count += 1
                    sum_denominator += denominator
                    sum_denominator_count += 1

                    _end = time.time()

                    print("traning model time:", _end - _start)

            # Search For Noise
            
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            idx = 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)

                if args.noise_after_transform:
                    pos_samples_1 = utils.train_diff_transform(pos_samples_1)
                    pos_samples_2 = utils.train_diff_transform(pos_samples_2)

                # Add Sample-wise Noise to each sample
                batch_noise, batch_start_idx = [], idx
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    sample_noise = random_noise[label[0].item()]
                    # c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                    # mask = np.zeros((c, h, w), np.float32)
                    # x1, x2, y1, y2 = mask_cord_list[idx]
                    if type(sample_noise) is np.ndarray:
                        sample_noise = torch.from_numpy(sample_noise).to(device)
                    else:
                        sample_noise = sample_noise.cpu().numpy()
                    # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    sample_noise = torch.from_numpy(sample_noise).to(device)
                    batch_noise.append(sample_noise)
                    idx += 1

                # Update sample-wise perturbation
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

                batch_noise = torch.stack(batch_noise).to(device)
                if args.attack_type == 'min-min':
                    if args.min_min_attack_fn == "eot_v1":
                        print('check eot_v1 right')
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform, eot_size=args.eot_size, one_gpu_eot_times=args.one_gpu_eot_times, cross_eot=args.cross_eot, pytorch_aug=args.pytorch_aug, dbindex_weight=args.dbindex_weight)
                        
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform, split_transform=args.split_transform)
                    else:
                        raise('Using wrong min_min_attack_fn in samplewise.')
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                for delta, label in zip(eta, labels):
                    # x1, x2, y1, y2 = mask_cord_list[label.item()]
                    # delta = delta[:, x1: x2, y1: y2]
                    if torch.is_tensor(random_noise):
                        random_noise[label[0].item()] = delta.detach().cpu().clone()
                    else:
                        random_noise[label[0].item()] = delta.detach().cpu().numpy()

                # print(torch.sum(random_noise_list[0] != random_noise_list[1]))
                # print(torch.sum(random_noise_list[0] != random_noise_list[2]))
                # print(torch.sum(random_noise_list[0] != random_noise_list[3]))
                # print(torch.sum(random_noise_list[1] != random_noise_list[2]))
                # print(torch.sum(random_noise_list[1] != random_noise_list[3]))
                # print(torch.sum(random_noise_list[2] != random_noise_list[3]))
                # # input()

                # print(torch.sum(random_noise))
                # input()

            noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255
                # print("noise_ave_value", noise_ave_value)

        # Here we save some samples in image.
        # if epoch_idx % 10 == 0 and not args.no_save:
        # # if True:
        #     if not os.path.exists('./images/'+save_name_pre):
        #         os.mkdir('./images/'+save_name_pre)
        #     images = []
        #     for group_idx in range(save_image_num):
        #         utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
        train_loss = sum_train_loss / float(sum_train_batch_size)
        numerator = sum_numerator / float(sum_numerator_count)
        denominator = sum_denominator / float(sum_denominator_count)
        print(train_loss)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
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

        results['numerator'].append(numerator)
        results['denominator'].append(denominator)

        # print("results['numerator']", results['numerator'])
        # print("results['denominator']", results['denominator'])

        # save statistics
        # print(len(results['best_loss']))
        # print(epoch_idx)
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save and args.local_rank == 0:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 3 == 0 and not args.no_save and args.local_rank == 0:
            torch.save(model.module.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            print("model saved at " + save_name_pre)

    if not args.no_save and args.local_rank == 0:
        torch.save(model.module.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

        piermaro_checkpoint = {'state_dict': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'perturbation': random_noise}
        torch.save(piermaro_checkpoint, 'results/{}_piermaro_model.pth'.format(save_name_pre))

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
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
    
    # transform_func = {'simclr': train_diff_transform, 
    #                   'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
    #                   'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
    #                   'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
    #                   'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
    #                   'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
    #                   'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
    #                   'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
    #                   }
    
    # if np.sum(args.augmentation_prob) == 0:
    #     my_transform_func = transform_func[args.augmentation]
    # else:
    #     my_transform_func = utils.train_diff_transform_prob(*args.augmentation_prob)

    # # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    # data_iter = iter(train_loader_simclr)
    
    # end_of_iteration = "END_OF_ITERATION"
    # total_top1, total_top5, total_num = 0.0, 0.0, 0.0
    # for j in range(0, args.train_step):
    #     try:
    #         next_item = next(data_iter, end_of_iteration)
    #         if next_item != end_of_iteration:
    #             (pos_samples_1, pos_samples_2, labels) = next_item
    #         else:
    #             del data_iter
    #             break
    #     except:
    #         raise('train loader iteration problem')

    #     pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
        
    #     target = torch.arange(0, pos_samples_1.shape[0]).to(device)

    #     model.eval()
    #     pos_samples_1 = my_transform_func(pos_samples_1)
    #     pos_samples_2 = my_transform_func(pos_samples_2)
    #     feature1, out1 = model(pos_samples_1)
    #     feature2, out2 = model(pos_samples_2)
    
    #     # compute cos similarity between each two groups of augmented samples ---> [B, B]
    #     sim_matrix = torch.mm(feature1, feature2.t())
    #     pos_sim = torch.sum(feature1 * feature2, dim=-1)
        
    #     # mask2 = (torch.ones_like(sim_matrix) - torch.eye(pos_samples_1.shape[0], device=sim_matrix.device)).bool()
    #     # # [B, B-1]
    #     # neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(pos_samples_1.shape[0], -1)
    #     # sim_weight, sim_indices = neg_sim_matrix2.topk(k=10, dim=-1)
        
    #     sim_indice_1 = sim_matrix.argsort(dim=0, descending=True) #[B, B]
    #     sim_indice_2 = sim_matrix.argsort(dim=1, descending=True) #[B, B]
    #     # print(sim_indice_1[0, :30])
    #     # print(sim_indice_2[:30, 0])

    #     total_top1 += torch.sum((sim_indice_1[:1, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #     total_top1 += torch.sum((sim_indice_2[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #     total_top5 += torch.sum((sim_indice_1[:5, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #     total_top5 += torch.sum((sim_indice_2[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
    #     total_num += pos_samples_1.shape[0] * 2
    
    # print(total_top1 / total_num, total_top5 / total_num, )

    return random_noise, save_name_pre


def plot_be(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, plot_input_data_loader):

    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1
            
    transform_func = {'simclr': train_diff_transform, 
                      'ReCrop_Hflip': utils.train_diff_transform_ReCrop_Hflip,
                      'ReCrop_Hflip_Bri': utils.train_diff_transform_ReCrop_Hflip_Bri,
                      'ReCrop_Hflip_Con': utils.train_diff_transform_ReCrop_Hflip_Con,
                      'ReCrop_Hflip_Sat': utils.train_diff_transform_ReCrop_Hflip_Sat,
                      'ReCrop_Hflip_Hue': utils.train_diff_transform_ReCrop_Hflip_Hue,
                      'Hflip_Bri': utils.train_diff_transform_Hflip_Bri,
                      'ReCrop_Bri': utils.train_diff_transform_ReCrop_Bri,
                      }

    if np.sum(args.augmentation_prob) == 0:
        my_transform_func = transform_func[args.augmentation]
    else:
        my_transform_func = utils.train_diff_transform_prob(*args.augmentation_prob)

    epochs = args.epochs
    print("The whole epochs are {}".format(epochs))
    if args.job_id == '':
        save_name_pre = 'unlearnable_plot_be_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_plot_be_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    
    feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []
    one_iter_for_plot_input = iter(plot_input_data_loader)
    plot_input_1, plot_input_2, plot_targets = next(one_iter_for_plot_input)
    if args.plot_be_mode == 'ave_augmentation':
        # print(type(plot_targets))
        # print(type(plot_targets[0]))
        # idx = np.where(plot_targets[:, 0] == 0)[0]
        plot_labels = plot_targets[:1024].to(device).cpu().numpy()
        # mnist_labels = plot_targets[:1024, 1].to(device).cpu().numpy()
        plot_input_1 = plot_input_1[:1024, :].to(device)
        # plot_input_2 = plot_input_2[:300:10, :].to(device)
        center_input = plot_input_1
        # plot_input_1 = train_diff_transform(center_input)
        # plot_input_2 = train_diff_transform(plot_input_2)
        plot_idx_color = None
        sample_num = 1024 # used in plot function
    
        if args.load_model_path != '':
            load_model_path = './results/{}.pth'.format(args.load_model_path)
            checkpoints = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoints)
        model.eval()
        feature_center, out_center = model(center_input)
        feature_center_bank.append(feature_center.cpu().detach().numpy())
        out_center_bank.append(out_center.cpu().detach().numpy())
        for i in range(40):
            plot_input_1 = my_transform_func(center_input)
            feature_1, out_1 = model(plot_input_1)
            feature1_bank.append(feature_1.cpu().detach().numpy())
            out1_bank.append(out_1.cpu().detach().numpy())
        feature1_bank = np.stack(feature1_bank, axis=0)
        out1_bank = np.stack(out1_bank, axis=0)
        feature1_bank = np.mean(feature1_bank, axis = 0)
        out1_bank = np.mean(out1_bank, axis = 0)
        
        if args.load_model_path2 != '':
            load_model_path = './results/{}.pth'.format(args.load_model_path2)
            checkpoints = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoints)
            model.eval()
            feature_center, out_center = model(center_input)
            feature_center_bank.append(feature_center.cpu().detach().numpy())
            out_center_bank.append(out_center.cpu().detach().numpy())
            for i in range(40):
                plot_input_2 = my_transform_func(center_input)
                feature_2, out_2 = model(plot_input_2)
                feature2_bank.append(feature_2.cpu().detach().numpy())
                out2_bank.append(out_2.cpu().detach().numpy())
            feature2_bank = np.stack(feature2_bank, axis=0)
            out2_bank = np.stack(out2_bank, axis=0)
            feature2_bank = np.mean(feature2_bank, axis = 0)
            out2_bank = np.mean(out2_bank, axis = 0)
        
        utils.plot_be([feature1_bank], [feature2_bank], feature_center_bank, plot_labels, args.load_model_path, sample_num, args.plot_be_mode, args.gray_test, args.augmentation, None, args.augmentation_prob,)
    
    if args.plot_be_mode == 'single_augmentation':
        # print(type(plot_targets))
        # print(type(plot_targets[0]))
        # idx = np.where(plot_targets[:, 0] == 0)[0]
        plot_labels = plot_targets[:1024].to(device).cpu().numpy()
        # mnist_labels = plot_targets[:1024, 1].to(device).cpu().numpy()
        plot_input_1 = plot_input_1[:1024, :].to(device)
        # plot_input_2 = plot_input_2[:300:10, :].to(device)
        center_input = plot_input_1
        # plot_input_1 = train_diff_transform(center_input)
        # plot_input_2 = train_diff_transform(plot_input_2)
        plot_idx_color = None
        sample_num = 1024 # used in plot function
    
        if args.load_model_path != '':
            load_model_path = './results/{}.pth'.format(args.load_model_path)
            checkpoints = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoints)
        model.eval()
        feature_center, out_center = model(center_input)
        feature_center_bank.append(feature_center.cpu().detach().numpy())
        out_center_bank.append(out_center.cpu().detach().numpy())
        for i in range(1):
            plot_input_1 = my_transform_func(center_input)
            feature_1, out_1 = model(plot_input_1)
            feature1_bank.append(feature_1.cpu().detach().numpy())
            out1_bank.append(out_1.cpu().detach().numpy())
        feature1_bank = np.stack(feature1_bank, axis=0)
        out1_bank = np.stack(out1_bank, axis=0)
        feature1_bank = np.mean(feature1_bank, axis = 0)
        out1_bank = np.mean(out1_bank, axis = 0)
        
        if args.load_model_path2 != '':
            load_model_path = './results/{}.pth'.format(args.load_model_path2)
            checkpoints = torch.load(load_model_path, map_location=device)
            model.load_state_dict(checkpoints)
            model.eval()
            feature_center, out_center = model(center_input)
            feature_center_bank.append(feature_center.cpu().detach().numpy())
            out_center_bank.append(out_center.cpu().detach().numpy())
            for i in range(1):
                plot_input_2 = my_transform_func(center_input)
                feature_2, out_2 = model(plot_input_2)
                feature2_bank.append(feature_2.cpu().detach().numpy())
                out2_bank.append(out_2.cpu().detach().numpy())
            feature2_bank = np.stack(feature2_bank, axis=0)
            out2_bank = np.stack(out2_bank, axis=0)
            feature2_bank = np.mean(feature2_bank, axis = 0)
            out2_bank = np.mean(out2_bank, axis = 0)
        
        utils.plot_be([feature1_bank], [feature2_bank], feature_center_bank, plot_labels, args.load_model_path, sample_num, args.plot_be_mode, args.gray_test, args.augmentation, None, args.augmentation_prob,)

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

    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size, kmeans_index=args.kmeans_index)
    train_data.replace_targets_with_id()
    if not args.org_label_noise and args.perturb_type == 'classwise':
        # we have to change the target randomly to give the noise a label
        train_data.replace_random_noise_class(random_noise_class)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    train_noise_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size, kmeans_index=args.kmeans_index)
    if not args.org_label_noise and args.perturb_type == 'classwise':
        train_noise_data.replace_random_noise_class(random_noise_class)
    train_noise_data.replace_targets_with_id()
    train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=args.num_workers, pin_memory=True)
    # test data don't have to change the target. by renjie3
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test, train_noise_after_transform=args.noise_after_transform, )
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test, train_noise_after_transform=args.noise_after_transform, )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    plot_input_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, gray=args.gray_test, class_4_train_size=args.class_4_train_size)
    plot_input_data_loader = DataLoader(plot_input_data, batch_size=1024, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                            num_steps=args.num_steps,
                                            step_size=args.step_size)

    # model setup and optimizer config
    model = Model(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size).cuda()
    # if args.no_bn:
    #     model = Model_no_bn(name='resnet18', head='mlp', feat_dim=feature_dim)
    # else:
    #     model = ParalellModel(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size)

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
            if args.perturb_type == 'samplewise':
                noise, save_name_pre = sample_wise_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, save_name_pre)

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
                    noise, save_name_pre = universal_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)

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

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    # for arg in vars(args):
    #     logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
