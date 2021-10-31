from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
from dataset import patch_noise_extend_to_img

import random
import matplotlib.pyplot as plt
import matplotlib

import kornia.augmentation as Kaug
import torch.nn as nn
import os
import pickle
from typing import Any, Callable, Optional, Tuple
import pandas as pd

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(CIFAR10Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
        with open(sampled_filepath, "rb") as f:
            sampled_data = pickle.load(f)
        if train:
            self.data = sampled_data["train_data"]
            self.targets = sampled_data["train_targets"]
        else:
            self.data = sampled_data["test_data"]
            self.targets = sampled_data["test_targets"]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_random_noise_class(self, random_noise_class):
        # print('length of targets is ', len(self.targets))
        # print(random_noise_class.shape)
        # for i in range(10):
        #     print(i, np.sum(random_noise_class == i))
        if len(self.targets) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.targets[i] = random_noise_class[i]
        else:
            raise('Replacing data noise class failed. Because the length is not consistent.')

    def add_noise_test_visualization(self, random_noise_class_test, noise):
        # print(noise.shape)
        # print(self.data[0][0][0])
        noise_255 = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)

        if len(self.data) == random_noise_class_test.shape[0]:
            for i in range(len(self.targets)):
                # print(noise_255[random_noise_class_test[i]].cpu().numpy()[0][0])
                # print(self.data[i][0][0])
                # print(self.targets[i], random_noise_class[i])
                self.data[i] += noise_255[random_noise_class_test[i]]
        else:
            raise('Add noise to data failed. Because the length is not consistent.')

        self.data = self.data.astype(np.uint8)


class SameImgCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super(SameImgCIFAR10Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
        with open(sampled_filepath, "rb") as f:
            sampled_data = pickle.load(f)
        if train:
            self.data = sampled_data["train_data"]
            self.targets = sampled_data["train_targets"]
        else:
            self.data = sampled_data["test_data"]
            self.targets = sampled_data["test_targets"]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_1, target

    def replace_random_noise_class(self, random_noise_class):
        # print('length of targets is ', len(self.targets))
        # print(random_noise_class.shape)
        # for i in range(10):
        #     print(i, np.sum(random_noise_class == i))
        if len(self.targets) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.targets[i] = random_noise_class[i]
        else:
            raise('Replacing data noise class failed. Because the length is not consistent.')

    def add_noise_test_visualization(self, random_noise_class_test, noise):
        print(type(noise))

        if len(self.data) == random_noise_class_test.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.data[i] += random_noise_class[random_noise_class_test[i]]
        else:
            raise('Add noise to data failed. Because the length is not consistent.')

class PoisonCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath='my_experiments/class_wise_cifar10/perturbation.pt'):
        super(PoisonCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
        with open(sampled_filepath, "rb") as f:
            sampled_data = pickle.load(f)
        if train:
            self.data = sampled_data["train_data"]
            self.targets = sampled_data["train_targets"]
        else:
            self.data = sampled_data["test_data"]
            self.targets = sampled_data["test_targets"]
        
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            noise = self.perturb_tensor[self.targets[idx]]
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)



    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_random_noise_class(self, random_noise_class):
        # print('length of targets is ', len(self.targets))
        # print(random_noise_class.shape)
        # for i in range(10):
        #     print(i, np.sum(random_noise_class == i))
        if len(self.targets) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.targets[i] = random_noise_class[i]
        else:
            raise('Replacing data noise class failed. Because the length is not consistent.')
    
    def add_noise_test_visualization(self, random_noise_class_test, noise):

        noise_255 = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        if len(self.data) == random_noise_class_test.shape[0]:
            for i in range(len(self.targets)):
                self.data[i] += noise_255[random_noise_class_test[i]]
        else:
            raise('Add noise to data failed. Because the length is not consistent.')

        self.data = self.data.astype(np.uint8)

class TransferCIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0):
        super(TransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
        with open(sampled_filepath, "rb") as f:
            sampled_data = pickle.load(f)
        if train:
            self.data = sampled_data["train_data"]
            self.targets = sampled_data["train_targets"]
        else:
            self.data = sampled_data["test_data"]
            self.targets = sampled_data["test_targets"]

        if perturb_tensor_filepath != None:
            self.perturb_tensor = torch.load(perturb_tensor_filepath)
            self.noise_255 = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None

        if random_noise_class_path != None:
            self.random_noise_class = np.load(random_noise_class_path)
        else:
            self.random_noise_class = None
        
        self.perturbation_budget = perturbation_budget

    # random_noise_class = np.load('noise_class_label.npy')
    #     self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
    #     self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
    #     self.data = self.data.astype(np.float32)
    #     for idx in range(len(self.data)):
    #         noise = self.perturb_tensor[self.targets[idx]]
    #         noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
    #         self.data[idx] = self.data[idx] + noise
    #         self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
    #     self.data = self.data.astype(np.uint8)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img.shape)
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            # print(self.perturb_tensor[self.random_noise_class[index]][0][0])
            # print("self.transform(img)", self.transform(img).shape)
            pos_1 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)
            pos_2 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

    def replace_random_noise_class(self, random_noise_class):
        # print('length of targets is ', len(self.targets))
        # print(random_noise_class.shape)
        # for i in range(10):
        #     print(i, np.sum(random_noise_class == i))
        if len(self.targets) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.targets[i] = random_noise_class[i]
        else:
            raise('Replacing data noise class failed. Because the length is not consistent.')
    
    def make_unlearnable(self, random_noise_class, noise):

        noise_255 = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        if len(self.data) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print("data:", self.data[i][0])
                # print("noise:", noise_255[random_noise_class[i]][0])
                self.data[i] += noise_255[random_noise_class[i]]
                # input()
            print("Making data unlearnable done")
        else:
            raise('Making data unlearnable failed. Because the length is not consistent.')

        self.data = self.data.astype(np.uint8)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

ToTensor_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_diff_transform = nn.Sequential(
    Kaug.RandomResizedCrop([32,32]),
    Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    Kaug.RandomGrayscale(p=0.2)
)

train_diff_transform2 = nn.Sequential(
    Kaug.RandomResizedCrop([32,32]),
    # Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    # Kaug.RandomGrayscale(p=0.2)
)

train_diff_transform3 = nn.Sequential(
    Kaug.RandomCrop([32,32], padding=4),
    Kaug.RandomHorizontalFlip(p=0.5),
)
    
def get_pairs_of_imgs(idx, clean_train_dataset, noise):
    clean_img = clean_train_dataset.data[idx]
    clean_img = transforms.functional.to_tensor(clean_img)
    unlearnable_img = torch.clamp(clean_img + noise[clean_train_dataset.targets[idx]], 0, 1)

    x = noise[clean_train_dataset.targets[idx]]
    x_min = torch.min(x)
    x_max = torch.max(x)
    noise_norm = (x - x_min) / (x_max - x_min)
    noise_norm = torch.clamp(noise_norm, 0, 1)

    return [clean_img, noise_norm, unlearnable_img]

def save_img_group(clean_train_dataset, noise, img_path):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    selected_idx = [random.randint(0, 1023) for _ in range(9)]
    img_grid = []
    for idx in selected_idx:
        img_grid += get_pairs_of_imgs(idx, clean_train_dataset, noise)

    img_grid_tensor = torchvision.utils.make_grid(torch.stack(img_grid), nrow=9, pad_value=255)
    npimg = img_grid_tensor.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(img_path)

def plot_loss(file_prename):
    pd_reader = pd.read_csv(file_prename+".csv")
    # print(pd_reader)

    epoch = pd_reader.values[:,0]
    loss = pd_reader.values[:,1]
    acc = pd_reader.values[:,2]

    fig, ax=plt.subplots(1,1,figsize=(9,6))
    ax1 = ax.twinx()

    p2 = ax.plot(epoch, loss,'r-', label = 'loss')
    ax.legend()
    p3 = ax1.plot(epoch,acc, 'b-', label = 'test_acc')
    ax1.legend()

    #显示图例
    # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
    # plt.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax1.set_ylabel('acc')
    plt.title('Training loss on generating model and clean test acc')
    plt.savefig(file_prename + ".png")