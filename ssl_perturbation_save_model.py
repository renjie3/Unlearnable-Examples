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
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise', 'clean_train', 'clean_train_softmax', 'plot_be', 'samplewise_myshuffle', 'samplewise_model_free', 'test_find_positive_pair'], help='Perturb type')
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
parser.add_argument('--plot_be_mode', default='ave_augmentation', type=str, choices=['ave_augmentation', 'augmentation', 'sample'], help='What samples to plot')
parser.add_argument('--plot_be_mode_feature', default='out', type=str, choices=['feature', 'out'], help='What to plot? Feature or out?')
parser.add_argument('--gray_train', default='no', type=str, choices=['gray', 'no', 'red'], help='gray_train')
# parser.add_argument('--gray_test', default='no', type=str, choices=['gray', 'no', 'red', 'gray_mnist', 'grayshift_mnist', 'colorshift_mnist', 'grayshift_font_mnist', 'grayshift2_font_mnist', 'grayshift_font_singledigit_mnist', 'grayshift_font_randomdigit_mnist', 'grayshiftlarge_font_randomdigit_mnist', 'grayshiftlarge_font_singldigit_mnist'], help='gray_test')
parser.add_argument('--gray_test', default='no', type=str, help='gray_test')
parser.add_argument('--augmentation', default='simclr', type=str, choices=['ReCrop_Hflip', 'simclr', 'ReCrop_Hflip_Bri', 'ReCrop_Hflip_Con', 'ReCrop_Hflip_Sat', 'ReCrop_Hflip_Hue', 'ReCrop_Bri', 'Hflip_Bri'], help='What')
parser.add_argument('--augmentation_prob', default=[0, 0, 0, 0], nargs='+', type=float, help='get augmentation by probility')
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
from model import Model
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from simclr import test_ssl, train_simclr, train_simclr_noise_return_loss_tensor, train_simclr_target_task, train_simclr_softmax, test_ssl_softmax
import random
import matplotlib.pyplot as plt
import matplotlib
from thop import profile, clever_format

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


def universal_perturbation_model_group(noise_generator, trainer, evaluator, model_group, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img):
    # Class-Wise perturbation
    # Generate Data loader

    condition = True
    # data_iter = iter(data_loader['train_dataset'])
    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_{}_{}_{}_{}'.format(args.job_id, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    model_group_schedule = [10, 200, 990]
    bilevel_train_step = model_group_schedule[0]
    perturbation_train_step = bilevel_train_step + model_group_schedule[1]
    model_train_step = perturbation_train_step + model_group_schedule[2]
    for epoch_idx in range(1, epochs+1):
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0,0
        # logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
        condition = 0
        for model in model_group:
            if hasattr(model, 'classify'):
                model.classify = True
        while condition < 2:
            if args.attack_type == 'min-min' and not args.load_model and (epoch_idx <= bilevel_train_step or epoch_idx >perturbation_train_step):
                print("training model parameters")
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    try:
                        next_item = next(data_iter, end_of_iteration)
                        if next_item != end_of_iteration:
                            (pos_samples_1, pos_samples_2, labels) = next_item
                            
                        else:
                            condition = 2
                            del data_iter
                            break
                        # print("condition", condition)
                    except:
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
                    for idx_model, model in enumerate(model_group):
                        model.train()
                        for param in model.parameters():
                            param.requires_grad = True
                        # trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)
                        batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer[idx_model], batch_size, temperature, noise_after_transform=args.noise_after_transform)
                        sum_train_loss += batch_train_loss
                        sum_train_batch_size += batch_size_count
                        break
                # model_train_step_count += 1
                train_loss = sum_train_loss / float(sum_train_batch_size)

            if epoch_idx <= perturbation_train_step:
                train_noise_loss_sum, train_noise_loss_count = 0, 0
                # print("check2")
                for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                # for i, (pos_samples_1, pos_samples_2, labels) in enumerate(train_noise_data_loader_simclr):
                    print(i, "one noise batch")
                    for idx_model in range(len(model_group)):
                        model_group[idx_model] = model_group[idx_model].to(device)
                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    # for idx_model, model in enumerate(model_group):
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
                    # print("go into group_model ", idx_model)
                    if epoch_idx <= bilevel_train_step:
                        model_group_step_size_schedule = args.step_size
                    else:
                        model_group_step_size_schedule = 0.4 / 255.0
                    if args.attack_type == 'min-min':
                        if args.min_min_attack_fn == "eot_v1":
                            _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                        elif args.min_min_attack_fn == "eot_v2":
                            _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v2(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                        elif args.min_min_attack_fn == "eot_v3":
                            _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v3(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                        elif args.min_min_attack_fn == "non_eot":
                            _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_model_group(pos_samples_1, pos_samples_2, labels, model_group, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, step_size_schedule=model_group_step_size_schedule)
                        elif args.min_min_attack_fn in ["pos/neg", "pos", "neg"]:
                            _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, target_task=args.min_min_attack_fn, noise_after_transform=args.noise_after_transform)
                        train_noise_loss_sum += train_noise_loss * pos_samples_1.shape[0]
                        train_noise_loss_count += 1
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
                print("train_noise_loss:", train_noise_loss_sum / float(train_noise_loss_count))
                condition += 1
            
        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
        # train_loss = sum_train_loss / float(sum_train_batch_size)
        # train_loss = train_noise_loss_sum / float(train_noise_loss_count)
        print(train_loss)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test_ssl(model_group[0], memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)

        if train_loss < best_loss:
            best_loss = train_loss
            best_loss_acc = test_acc_1
            if not args.no_save:
                torch.save(model_group[0].state_dict(), 'results/{}_model.pth'.format(save_name_pre))
                torch.save(random_noise, 'results/{}_bestloss_perturbation.pt'.format(save_name_pre))
        results['best_loss'].append(best_loss)
        results['best_loss_acc'].append(best_loss_acc)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 10 == 0 and not args.no_save:
            torch.save(model_group[0].state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            print("model saved at " + save_name_pre)
    
    if not args.no_save:
        torch.save(model_group[0].state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
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


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img):
    # datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
    #                                               eval_batch_size=args.eval_batch_size,
    #                                               train_data_type=args.train_data_type,
    #                                               train_data_path=args.train_data_path,
    #                                               test_data_type=args.test_data_type,
    #                                               test_data_path=args.test_data_path,
    #                                               num_of_workers=args.num_of_workers,
    #                                               seed=args.seed, no_train_augments=True)

    # if args.train_data_type == 'ImageNetMini' and args.perturb_type == 'samplewise':
    #     data_loader = datasets_generator._split_validation_set(0.2, train_shuffle=False, train_drop_last=False)
    #     data_loader['train_dataset'] = data_loader['train_subset']
    # else:
    #     data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            if args.shuffle_train_perturb_data:
                noise = random_noise[label.item()]
            else:
                noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_samplewise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_samplewise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    for epoch_idx in range(1, epochs+1):
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
                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        if args.shuffle_train_perturb_data:
                            sample_noise = random_noise[label.item()]
                        else:
                            sample_noise = random_noise[train_idx]
                        c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                        mask = np.zeros((c, h, w), np.float32)
                        x1, x2, y1, y2 = mask_cord_list[train_idx]
                        if type(sample_noise) is np.ndarray:
                            mask[:, x1: x2, y1: y2] = sample_noise
                        else:
                            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        sample_noise = torch.from_numpy(mask).to(device)
                        # images[i] = images[i] + sample_noise
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)
                        train_idx += 1

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
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

            # Search For Noise
            
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            idx = 0
            for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)

                # Add Sample-wise Noise to each sample
                batch_noise, batch_start_idx = [], idx
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    sample_noise = random_noise[idx]
                    c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
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
                    if args.min_min_attack_fn == "eot_v1":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform)
                    else:
                        raise('Using wrong min_min_attack_fn in samplewise.')
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                for i, delta in enumerate(eta):
                    x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                    delta = delta[:, x1: x2, y1: y2]
                    if torch.is_tensor(random_noise):
                        random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                    else:
                        random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

                noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255
                # print("noise_ave_value", noise_ave_value)

        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
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

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise, save_name_pre
    else:
        return random_noise, save_name_pre
    
def sample_wise_model_free_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img):
    # still working on it. need a feature extractor. maybe PCA
    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            if args.shuffle_train_perturb_data:
                noise = random_noise[label.item()]
            else:
                noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'best_loss': [], 'noise_ave_value': []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_samplewise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_samplewise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    for epoch_idx in range(1, epochs+1):
        train_idx = 0
        condition = True
            
        train_noise_loss_sum, train_noise_loss_count = 0, 0
        idx = 0
        for i, (pos_samples_1, pos_samples_2, labels) in tqdm(enumerate(train_noise_data_loader_simclr), total=len(train_noise_data_loader_simclr), desc="Training images"):
            pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)

            # Add Sample-wise Noise to each sample
            batch_noise, batch_start_idx = [], idx
            for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                sample_noise = random_noise[idx]
                c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
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
            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                if args.min_min_attack_fn in ["eot_v1", "eot_v1_pos", "eot_v1_neg"]:
                    _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1_model_free(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                elif args.min_min_attack_fn in ["non_eot", "pos", "neg"]:
                    _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_model_free(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform)
                else:
                    raise('Using wrong min_min_attack_fn in samplewise.')
            # elif args.attack_type == 'min-max':
            #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise('Invalid attack')
            train_noise_loss_sum += train_noise_loss
            train_noise_loss_count += 1

            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                if torch.is_tensor(random_noise):
                    random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

            noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255

        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
        train_loss = train_noise_loss_sum / float(train_noise_loss_count)
        print(train_loss)
        results['train_loss'].append(train_loss)
        results['noise_ave_value'].append(noise_ave_value)

        if train_loss < best_loss:
            best_loss = train_loss
        results['best_loss'].append(best_loss)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        if epoch_idx % 10 == 0 and not args.no_save:
            torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
            print("noise saved at " + save_name_pre)

    if not args.no_save:
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise, save_name_pre
    else:
        return random_noise, save_name_pre


def sample_wise_perturbation_myshuffle(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, train_dataset):
    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_dataset:
        noise = random_noise[idx]
        mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos_samples_1.shape, patch_location=args.patch_location)
        mask_cord_list.append(mask_cord)
        idx += 1

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], 'noise_ave_value': [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_samplewise_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_samplewise_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    for epoch_idx in range(1, epochs+1):
        train_idx = 0
        condition = True
        if (epoch_idx-1) % args.shuffle_step == 0:
            shuffle_idx = np.random.permutation(1024).reshape((2, batch_size))
        batch_idx_iter = iter(shuffle_idx)
        # data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0,0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0
        while condition:
            if args.attack_type == 'min-min' and not args.load_model and not args.perturb_first:
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    try:
                    # if True:
                        next_item = next(batch_idx_iter, end_of_iteration)
                        if next_item != end_of_iteration:
                            pos_samples_1 = []
                            pos_samples_2 = []
                            labels = []
                            for s_idx in next_item:
                                pos_1, pos_2, label = train_dataset[s_idx]
                                pos_samples_1.append(pos_1)
                                pos_samples_2.append(pos_2)
                                labels.append(label)
                            pos_samples_1 = torch.stack(pos_samples_1)
                            pos_samples_2 = torch.stack(pos_samples_2)
                            labels = torch.tensor(labels)
                        else:
                            condition = False
                            del batch_idx_iter
                            break
                    except:
                        # data_iter = iter(data_loader['train_dataset'])
                        # (pos_1, pos_2, labels) = next(data_iter)
                        raise('train loader iteration problem')

                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        sample_noise = random_noise[label.item()]
                        c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                        mask = np.zeros((c, h, w), np.float32)
                        x1, x2, y1, y2 = mask_cord_list[label.item()]
                        if type(sample_noise) is np.ndarray:
                            mask[:, x1: x2, y1: y2] = sample_noise
                        else:
                            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        sample_noise = torch.from_numpy(mask).to(device)
                        # images[i] = images[i] + sample_noise
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
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

            # Search For Noise
            
            train_noise_loss_sum, train_noise_loss_count = 0, 0
            idx = 0
            for i, batch_s_idx in tqdm(enumerate(shuffle_idx), total=len(shuffle_idx), desc="Training images"):

                # manually construct the batch of images
                pos_samples_1 = []
                pos_samples_2 = []
                labels = []
                for s_idx in batch_s_idx:
                    pos_1, pos_2, label = train_dataset[s_idx]
                    pos_samples_1.append(pos_1)
                    pos_samples_2.append(pos_2)
                    labels.append(label)
                pos_samples_1 = torch.stack(pos_samples_1)
                pos_samples_2 = torch.stack(pos_samples_2)
                labels = torch.tensor(labels)

                pos_samples_1, pos_samples_2, labels, model = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device), model.to(device)

                # Add Sample-wise Noise to each sample
                batch_noise, batch_start_idx = [], idx
                for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                    sample_noise = random_noise[idx]
                    c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
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
                    if args.min_min_attack_fn == "eot_v1":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor_eot_v1(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug)
                    elif args.min_min_attack_fn == "non_eot":
                        _, eta, train_noise_loss = noise_generator.min_min_attack_simclr_return_loss_tensor(pos_samples_1, pos_samples_2, labels, model, optimizer, None, random_noise=batch_noise, batch_size=batch_size, temperature=temperature, flag_strong_aug=args.strong_aug, noise_after_transform=args.noise_after_transform)
                    else:
                        raise('Using wrong min_min_attack_fn in samplewise.')
                # elif args.attack_type == 'min-max':
                #     perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
                else:
                    raise('Invalid attack')

                for i, delta in enumerate(eta):
                    x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                    delta = delta[:, x1: x2, y1: y2]
                    if torch.is_tensor(random_noise):
                        random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                    else:
                        random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

                noise_ave_value = np.mean(np.absolute(random_noise.to('cpu').numpy())) * 255
                # print("noise_ave_value", noise_ave_value)

            if args.attack_type == 'min-min' and not args.load_model and args.perturb_first:
                # Train Batch for min-min noise
                end_of_iteration = "END_OF_ITERATION"
                for j in range(0, args.train_step):
                    try:
                    # if True:
                        next_item = next(batch_idx_iter, end_of_iteration)
                        if next_item != end_of_iteration:
                            pos_samples_1 = []
                            pos_samples_2 = []
                            labels = []
                            for s_idx in next_item:
                                pos_1, pos_2, label = train_dataset[s_idx]
                                pos_samples_1.append(pos_1)
                                pos_samples_2.append(pos_2)
                                labels.append(label)
                            pos_samples_1 = torch.stack(pos_samples_1)
                            pos_samples_2 = torch.stack(pos_samples_2)
                            labels = torch.tensor(labels)
                        else:
                            condition = False
                            del batch_idx_iter
                            break
                    except:
                        # data_iter = iter(data_loader['train_dataset'])
                        # (pos_1, pos_2, labels) = next(data_iter)
                        raise('train loader iteration problem')

                    pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
                    # Add Sample-wise Noise to each sample
                    train_pos_1 = []
                    train_pos_2 = []
                    for i, (pos_1, pos_2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
                        sample_noise = random_noise[label.item()]
                        c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
                        mask = np.zeros((c, h, w), np.float32)
                        x1, x2, y1, y2 = mask_cord_list[label.item()]
                        if type(sample_noise) is np.ndarray:
                            mask[:, x1: x2, y1: y2] = sample_noise
                        else:
                            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        # mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        sample_noise = torch.from_numpy(mask).to(device)
                        # images[i] = images[i] + sample_noise
                        train_pos_1.append(pos_samples_1[i]+sample_noise)
                        train_pos_2.append(pos_samples_2[i]+sample_noise)

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, torch.stack(train_pos_1).to(device), torch.stack(train_pos_2).to(device), optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform)
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

        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
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

    # Update Random Noise to shape
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise, save_name_pre
    else:
        return random_noise, save_name_pre


def clean_train(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, plot_input_data_loader):

    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_cleantrain_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_cleantrain_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])
    
    if args.plot_process:
    
        feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []
        one_iter_for_plot_input = iter(plot_input_data_loader)
        plot_input_1, plot_input_2, plot_labels = next(one_iter_for_plot_input)
        if args.plot_process_mode == 'pair':
            plot_labels = plot_labels[:300:10].to(device).cpu().numpy()
            plot_input_1 = plot_input_1[:300:10, :].to(device)
            plot_input_2 = plot_input_2[:300:10, :].to(device)
            center_input = plot_input_1
            plot_input_1 = train_diff_transform(plot_input_1)
            plot_input_2 = train_diff_transform(plot_input_2)
            plot_idx_color = None
            sample_num = 30
        elif args.plot_process_mode == 'augmentation':
            plot_labels = plot_labels[:50:10].to(device)
            plot_input_1 = plot_input_1[:50:10, :].to(device)
            plot_input_2 = plot_input_2[:50:10, :].to(device)
            augment_samples_1 = []
            augment_samples_2 = []
            augment_labels = []
            augment_idx_color = []
            center_input = plot_input_1
            center_labels = plot_labels
            print("center label: ", center_labels)
            idx_color = torch.tensor([0,1,2,3,4]).to(device)
            for i in range(6):
                augment_samples_1.append(train_diff_transform(plot_input_1))
                augment_samples_2.append(train_diff_transform(plot_input_2))
                augment_labels.append(plot_labels)
                augment_idx_color.append(idx_color)
            augment_labels.append(plot_labels)
            augment_idx_color.append(idx_color)
            plot_input_1 = torch.cat(augment_samples_1, dim=0).contiguous()
            plot_input_2 = torch.cat(augment_samples_2, dim=0).contiguous()
            plot_labels = torch.cat(augment_labels, dim=0).contiguous().cpu().numpy()
            plot_idx_color = torch.cat(augment_idx_color, dim=0).contiguous().cpu().numpy()
            print(plot_input_1.shape)
            print(plot_input_2.shape)
            print(plot_labels.shape)
            sample_num = 30
        elif args.plot_process_mode == 'center':
            plot_labels = plot_labels[:60].to(device).cpu().numpy()
            all_the_input = plot_input_1[:60, :].to(device)
            plot_input_1 = all_the_input[:20, :].to(device)
            plot_input_2 = all_the_input[20:40, :].to(device)
            center_input = all_the_input[40:60, :].to(device)
            plot_idx_color = None
            print(plot_input_1.shape)
            print(plot_input_2.shape)
            print(plot_labels.shape)
            sample_num = 20

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    for epoch_idx in range(1, epochs+1):
        train_idx = 0
        condition = True
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0, 0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0
        
        if args.plot_process:
            model.eval()
            feature_1, out_1 = model(plot_input_1)
            feature_2, out_2 = model(plot_input_2)
            feature_center, out_center = model(center_input)
            feature1_bank.append(feature_1.cpu().detach().numpy())
            feature2_bank.append(feature_2.cpu().detach().numpy())
            feature_center_bank.append(feature_center.cpu().detach().numpy())
            out1_bank.append(out_1.cpu().detach().numpy())
            out2_bank.append(out_2.cpu().detach().numpy())
            out_center_bank.append(out_center.cpu().detach().numpy())
        
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

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    if len(args.num_den_sheduler) == 1:
                        if args.min_min_attack_fn in ["pos/neg", "pos", "neg"]:
                            batch_train_loss, batch_size_count, numerator, denominator = train_simclr_target_task(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform, target_task=args.min_min_attack_fn)
                        else:
                            batch_train_loss, batch_size_count, numerator, denominator = train_simclr(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform, mix=args.mix, augmentation=args.augmentation, augmentation_prob=args.augmentation_prob)
                    elif len(args.num_den_sheduler) == 2:
                        num_den_sheduler_fn = []
                        while len(num_den_sheduler_fn) <= epochs + 2:
                            num_den_sheduler_fn += ['pos' for i in range(args.num_den_sheduler[0])]
                            num_den_sheduler_fn += ['neg' for i in range(args.num_den_sheduler[1])]
                        batch_train_loss, batch_size_count, numerator, denominator = train_simclr_target_task(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform, target_task=num_den_sheduler_fn[epoch_idx])
                    else:
                        raise("Wrong num_den_sheduler")

                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count
                    sum_numerator += numerator
                    sum_numerator_count += 1
                    sum_denominator += denominator
                    sum_denominator_count += 1
            else:
                condition = False

        # Here we plot the process
        if epoch_idx <= 100:
            if epoch_idx % 10 == 0 and args.plot_process:
                if args.plot_process_feature == 'out':
                    utils.plot_process(out1_bank, out2_bank, out_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 10)
                elif args.plot_process_feature == 'feature':
                    utils.plot_process(feature1_bank, feature2_bank, feature_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 10)
                feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []
        else:
            if epoch_idx % 50 == 0 and args.plot_process:
                if args.plot_process_feature == 'out':
                    utils.plot_process(out1_bank, out2_bank, out_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 50)
                elif args.plot_process_feature == 'feature':
                    utils.plot_process(feature1_bank, feature2_bank, feature_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 50)
                feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []

        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
        
        test_acc_1, test_acc_5 = test_ssl(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        if not args.just_test:
            train_loss = sum_train_loss / float(sum_train_batch_size)
            numerator = sum_numerator / float(sum_numerator_count)
            denominator = sum_denominator / float(sum_denominator_count)
            results['train_loss'].append(train_loss)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)

            results['numerator'].append(numerator)
            results['denominator'].append(denominator)

            if train_loss < best_loss:
                best_loss = train_loss
                best_loss_acc = test_acc_1
                if not args.no_save:
                    torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            results['best_loss'].append(best_loss)
            results['best_loss_acc'].append(best_loss_acc)

            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
            if not args.no_save:
                data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

            if epoch_idx % 10 == 0 and not args.no_save:
                torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
                torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
                print("model saved at " + save_name_pre)
        else:
            break

    if not args.no_save and not args.just_test:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

    # Update Random Noise to shape
    # if torch.is_tensor(random_noise):
    #     new_random_noise = []
    #     for idx in range(len(random_noise)):
    #         sample_noise = random_noise[idx]
    #         c, h, w = pos_1.shape[0], pos_1.shape[1], pos_1.shape[2]
    #         mask = np.zeros((c, h, w), np.float32)
    #         x1, x2, y1, y2 = mask_cord_list[idx]
    #         mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
    #         new_random_noise.append(torch.from_numpy(mask))
    #     new_random_noise = torch.stack(new_random_noise)
    #     return new_random_noise, save_name_pre
    # else:
    return random_noise, save_name_pre

def clean_train_softmax(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, plot_input_data_loader):

    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    epochs = args.epochs
    save_image_num = args.save_image_num
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [],}
    if args.job_id == '':
        save_name_pre = 'unlearnable_cleantrainsoftmax_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_cleantrainsoftmax_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_loss = 10000000
    best_loss_acc = 0
    # data_iter = iter(data_loader['train_dataset'])

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    for epoch_idx in range(1, epochs+1):
        train_idx = 0
        condition = True
        data_iter = iter(train_loader_simclr)
        sum_train_loss, sum_train_batch_size = 0, 0
        sum_numerator, sum_numerator_count = 0, 0
        sum_denominator, sum_denominator_count = 0, 0
        
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

                    model.train()
                    for param in model.parameters():
                        param.requires_grad = True
                    batch_train_loss, batch_size_count = train_simclr_softmax(model, pos_samples_1, pos_samples_2, optimizer, batch_size, temperature, noise_after_transform=args.noise_after_transform, mix=args.mix, augmentation=args.augmentation, augmentation_prob=args.augmentation_prob)

                    sum_train_loss += batch_train_loss
                    sum_train_batch_size += batch_size_count
            else:
                condition = False

        # Here we plot the process
        if epoch_idx <= 100:
            if epoch_idx % 10 == 0 and args.plot_process:
                if args.plot_process_feature == 'out':
                    utils.plot_process(out1_bank, out2_bank, out_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 10)
                elif args.plot_process_feature == 'feature':
                    utils.plot_process(feature1_bank, feature2_bank, feature_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 10)
                feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []
        else:
            if epoch_idx % 50 == 0 and args.plot_process:
                if args.plot_process_feature == 'out':
                    utils.plot_process(out1_bank, out2_bank, out_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 50)
                elif args.plot_process_feature == 'feature':
                    utils.plot_process(feature1_bank, feature2_bank, feature_center_bank, plot_labels, save_name_pre, epoch_idx, sample_num, args.plot_process_mode, plot_idx_color, 50)
                feature1_bank, feature2_bank, feature_center_bank, out1_bank, out2_bank, out_center_bank = [], [], [], [], [], []

        # Here we save some samples in image.
        if epoch_idx % 10 == 0 and not args.no_save:
        # if True:
            if not os.path.exists('./images/'+save_name_pre):
                os.mkdir('./images/'+save_name_pre)
            images = []
            for group_idx in range(save_image_num):
                utils.save_img_group(train_data_for_save_img, random_noise, './images/{}/{}.png'.format(save_name_pre, group_idx))
        
        
        test_acc_1, test_acc_5 = test_ssl_softmax(model, memory_loader, test_loader, k, temperature, epoch_idx, epochs)
        if not args.just_test:
            train_loss = sum_train_loss / float(sum_train_batch_size)
            results['train_loss'].append(train_loss)
            results['test_acc@1'].append(test_acc_1)
            results['test_acc@5'].append(test_acc_5)

            if train_loss < best_loss:
                best_loss = train_loss
                best_loss_acc = test_acc_1
                if not args.no_save:
                    torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
            results['best_loss'].append(best_loss)
            results['best_loss_acc'].append(best_loss_acc)

            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, epoch_idx + 1))
            if not args.no_save:
                data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

            if epoch_idx % 10 == 0 and not args.no_save:
                torch.save(model.state_dict(), 'results/{}_checkpoint_model.pth'.format(save_name_pre))
                torch.save(random_noise, 'results/{}_checkpoint_perturbation.pt'.format(save_name_pre))
                print("model saved at " + save_name_pre)
        else:
            break

    if not args.no_save and not args.just_test:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))
        utils.plot_loss('./results/{}_statistics'.format(save_name_pre))

    return random_noise, save_name_pre

def test_find_positive_pair(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, train_loader_simclr, train_noise_data_loader_simclr, batch_size, temperature, memory_loader, test_loader, k, train_data_for_save_img, plot_input_data_loader):

    mask_cord_list = []
    idx = 0
    for pos_samples_1, pos_samples_2, labels in train_loader_simclr:
        for i, (pos1, pos2, label) in enumerate(zip(pos_samples_1, pos_samples_2, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=pos1.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    epochs = args.epochs
    print("The whole epochs are {}".format(epochs))
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_loss': [], "best_loss_acc": [], "numerator": [], "denominator": []}
    if args.job_id == '':
        save_name_pre = 'unlearnable_testFindPositivePair_local_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    else:
        save_name_pre = 'unlearnable_testFindPositivePair_{}_{}_{}_{}_{}'.format(args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    # data_iter = iter(data_loader['train_dataset'])
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

    # logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    data_iter = iter(train_loader_simclr)
    
    end_of_iteration = "END_OF_ITERATION"
    total_top1, total_top5, total_num = 0.0, 0.0, 0.0
    for j in range(0, args.train_step):
        try:
            next_item = next(data_iter, end_of_iteration)
            if next_item != end_of_iteration:
                (pos_samples_1, pos_samples_2, labels) = next_item
            else:
                del data_iter
                break
        except:
            raise('train loader iteration problem')

        pos_samples_1, pos_samples_2, labels = pos_samples_1.to(device), pos_samples_2.to(device), labels.to(device)
        
        target = torch.arange(0, pos_samples_1.shape[0]).to(device)

        model.eval()
        pos_samples_1 = my_transform_func(pos_samples_1)
        pos_samples_2 = my_transform_func(pos_samples_2)
        feature1, out1 = model(pos_samples_1)
        feature2, out2 = model(pos_samples_2)
    
        # compute cos similarity between each two groups of augmented samples ---> [B, B]
        sim_matrix = torch.mm(feature1, feature2.t())
        sim_indice_1 = sim_matrix.argsort(dim=0, descending=True) #[B, B]
        sim_indice_2 = sim_matrix.argsort(dim=1, descending=True) #[B, B]
        # print(sim_indice_1[0, :30])
        # print(sim_indice_2[:30, 0])

        total_top1 += torch.sum((sim_indice_1[:1, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_top1 += torch.sum((sim_indice_2[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_top5 += torch.sum((sim_indice_1[:5, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_top5 += torch.sum((sim_indice_2[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        total_num += pos_samples_1.shape[0] * 2

    return total_top1 / total_num * 100, total_top5 / total_num * 100

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
            plot_input_1 = transform_func[args.augmentation](center_input)
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
            plot_input_2 = transform_func[args.augmentation](center_input)
            feature_2, out_2 = model(plot_input_2)
            feature2_bank.append(feature_2.cpu().detach().numpy())
            out2_bank.append(out_2.cpu().detach().numpy())
        feature2_bank = np.stack(feature2_bank, axis=0)
        out2_bank = np.stack(out2_bank, axis=0)
        feature2_bank = np.mean(feature2_bank, axis = 0)
        out2_bank = np.mean(out2_bank, axis = 0)
        
        utils.plot_be([feature1_bank], [feature2_bank], feature_center_bank, plot_labels, args.load_model_path, sample_num, args.plot_be_mode, args.gray_test, args.augmentation, None)

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

    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'curren_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    # model = Model(feature_dim, arch=args.arch).cuda()
    # for m in model.parameters():
    #     test1 = m
    #     break

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
    # if args.noise_shape[0] == 10:
    #     random_noise_class = np.load('noise_class_label.npy')
    # else:
    #     random_noise_class = np.load('noise_class_label_' + str(args.noise_shape[0]) + 'class.npy')
    if args.class_4:
        random_noise_class = np.load('noise_class_label_1024_4class.npy')
    else:
        random_noise_class = np.load('noise_class_label.npy')
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size)
    if not args.org_label_noise and args.perturb_type == 'classwise':
        # we have to change the target randomly to give the noise a label
        train_data.replace_random_noise_class(random_noise_class)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=2, pin_memory=True, drop_last=True)
    train_noise_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, mix=args.mix, gray=args.gray_train, class_4_train_size=args.class_4_train_size)
    if not args.org_label_noise and args.perturb_type == 'classwise':
        train_noise_data.replace_random_noise_class(random_noise_class)
    if args.shuffle_train_perturb_data:
        train_noise_data.replace_targets_with_id()
    if args.perturb_type == 'samplewise_myshuffle':
        train_data.replace_targets_with_id()
    train_noise_data_loader = DataLoader(train_noise_data, batch_size=batch_size, shuffle=args.shuffle_train_perturb_data, num_workers=2, pin_memory=True)
    # test data don't have to change the target. by renjie3
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True, class_4=args.class_4, mix=args.mix, gray=args.gray_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    plot_input_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, class_4=args.class_4, train_noise_after_transform=args.noise_after_transform, gray=args.gray_test, class_4_train_size=args.class_4_train_size)
    plot_input_data_loader = DataLoader(plot_input_data, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    # model setup and optimizer config
    if args.model_group > 1:
        model = [Model(feature_dim, arch=args.arch).cuda() for _ in range(args.model_group)]
        optimizer = [optim.Adam(model[i].parameters(), lr=1e-3, weight_decay=1e-6) for i in range(args.model_group)]
    else:
        model = Model(feature_dim, arch=args.arch, train_mode=args.perturb_type, f_logits_dim=args.batch_size).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    c = len(memory_data.classes)
    
    if args.load_model:
        # unlearnable_cleantrain_41501264_1_20211204151414_0.5_512_1000_final_model
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        model.load_state_dict(checkpoints)
        # ENV = checkpoint['ENV']
        # trainer.global_step = ENV['global_step']
        # logger.info("File %s loaded!" % (load_model_path))

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
            if args.perturb_type == 'samplewise':
                noise, save_name_pre = sample_wise_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)
                
            elif args.perturb_type == 'samplewise_model_free':
                noise, save_name_pre = sample_wise_model_free_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)

            elif args.perturb_type == 'samplewise_myshuffle':
                noise, save_name_pre = sample_wise_perturbation_myshuffle(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, train_data)

            elif args.perturb_type == 'clean_train':
                noise, save_name_pre = clean_train(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'clean_train_softmax':
                noise, save_name_pre = clean_train_softmax(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'plot_be':
                noise, save_name_pre = plot_be(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                
            elif args.perturb_type == 'test_find_positive_pair':
                top1_acc, top5_acc = test_find_positive_pair(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data, plot_input_data_loader)
                print("The test top1 acc is {}. \n The test top5 acc is {}.".format(top1_acc, top5_acc))
                
            elif args.perturb_type == 'classwise':
                # noise = universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV)
                if args.model_group > 1:
                    noise, save_name_pre = universal_perturbation_model_group(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)
                else:
                    noise, save_name_pre = universal_perturbation(noise_generator, None, None, model, None, optimizer, None, random_noise, ENV, train_loader, train_noise_data_loader, batch_size, temperature, memory_loader, test_loader, k, train_data)
            if not args.no_save:
                torch.save(noise, 'results/{}perturbation.pt'.format(save_name_pre))
                # logger.info(noise)
                logger.info(noise.shape)
                logger.info('Noise saved at %s' % 'results/{}perturbation.pt'.format(save_name_pre))
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
