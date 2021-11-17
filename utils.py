from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
# from dataset import patch_noise_extend_to_img

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

ToTensor_transform = transforms.Compose([
    transforms.ToTensor(),
])

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

def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask

class RandomLabelCIFAR10(CIFAR10):
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

        super(RandomLabelCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        if train:
            random_noise_class = np.load("/mnt/home/renjie3/Documents/unlearnable/Unlearnable-Examples/noise_class_label.npy")
        else:
            random_noise_class = np.load("/mnt/home/renjie3/Documents/unlearnable/Unlearnable-Examples/noise_class_label_test.npy")
            # my_perturb = torch.load("my_experiments/class_wise_cifar10_diff_simclr_aug/perturbation.pt")

            # noise_255 = my_perturb.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()

            # self.data = self.data.astype(np.float32)

            # if len(self.data) == random_noise_class.shape[0]:
            #     for i in range(len(self.targets)):
            #         self.data[i] += noise_255[random_noise_class[i]]
            #         self.data[i] = np.clip(self.data[i], a_min=0, a_max=255)
            # else:
            #     raise('Add noise to data failed. Because the length is not consistent.')

            # self.data = self.data.astype(np.uint8)

        
        if len(self.targets) == random_noise_class.shape[0]:
            for i in range(len(self.targets)):
                # print(self.targets[i], random_noise_class[i])
                self.targets[i] = random_noise_class[i]
        else:
            raise('Replacing data noise class failed. Because the length is not consistent.')

class SampledCIFAR10(CIFAR10):
    """Sample 4 class * 256 pictures from CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        class_4: bool = True,
    ) -> None:

        super(SampledCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if class_4:
            sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
            with open(sampled_filepath, "rb") as f:
                sampled_data = pickle.load(f)
            if train:
                self.data = sampled_data["train_data"]
                self.targets = sampled_data["train_targets"]
            else:
                self.data = sampled_data["test_data"]
                self.targets = sampled_data["test_targets"]

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
        class_4: bool = True,
        train_noise_after_transform: bool = True,
    ) -> None:

        super(CIFAR10Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if class_4:
            sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
            with open(sampled_filepath, "rb") as f:
                sampled_data = pickle.load(f)
            if train:
                self.data = sampled_data["train_data"]
                self.targets = sampled_data["train_targets"]
            else:
                self.data = sampled_data["test_data"]
                self.targets = sampled_data["test_targets"]
        self.train_noise_after_transform = train_noise_after_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            if self.train_noise_after_transform:
                pos_1 = train_transform(img)
                pos_2 = train_transform(img)
            else:
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

    def replace_targets_with_id(self):
        for i in range(len(self.targets)):
            # print(self.targets[i], random_noise_class[i])
            self.targets[i] = i

    def add_noise_test_visualization(self, random_noise_class_test, noise):
        # print(noise.shape)
        # print(self.data[0][0][0])
        noise_255 = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        org_data = self.data
        # print(org_data[0])
        self.data = self.data.astype(np.float32)

        if len(self.data) == random_noise_class_test.shape[0]:
            for i in range(len(self.targets)):
                # print(noise_255[random_noise_class_test[i]][0][0])
                # print(self.data[i][0][0])
                # print(self.targets[i], random_noise_class[i])
                self.data[i] += noise_255[random_noise_class_test[i]]
                self.data[i] = np.clip(self.data[i], a_min=0, a_max=255)
        else:
            raise('Add noise to data failed. Because the length is not consistent.')

        self.data = self.data.astype(np.uint8)
        # print("org_data[0]", org_data[0,0,28])
        # print("self.data[0]", self.data[0,0,28])
        # for i in range(len(org_data[0,0])):
        #     print(i, np.mean(org_data[0,0,i] - self.data[0,0,i]))
        # print(np.mean(org_data.astype(np.float32) - self.data.astype(np.float32)))
        # input()

        # self.data = self.data.astype(np.float32)
        # for idx in range(len(self.data)):
        #     noise = self.noise_255[self.targets[idx]]
        #     noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
        #     self.data[idx] = self.data[idx] + noise
        #     self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        # self.data = self.data.astype(np.uint8)


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
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
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

        noise_255 = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
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
    def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0, class_4: bool = True, samplewise_perturb: bool = False, org_label_flag: bool = False):
        super(TransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

        if class_4:
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
            self.noise_255 = self.perturb_tensor.mul(255*perturbation_budget).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = None

        if random_noise_class_path != None:
            self.random_noise_class = np.load(random_noise_class_path)
        else:
            self.random_noise_class = None
        
        self.perturbation_budget = perturbation_budget

    # random_noise_class = np.load('noise_class_label.npy')
        # self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        self.data = self.data.astype(np.float32)
        for idx in range(len(self.data)):
            if not samplewise_perturb:
                if org_label_flag:
                    noise = self.noise_255[self.targets[idx]]
                else:
                    noise = self.noise_255[self.random_noise_class[idx]]
            else:
                noise = self.noise_255[idx]
                # print("check it goes samplewise.")
            noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # print(img[0][0])
        img = Image.fromarray(img)
        # print("np.shape(img)", np.shape(img))

        if self.transform is not None:
            # print(self.perturb_tensor[self.random_noise_class[index]][0][0])
            # print("self.transform(img)", self.transform(img).shape)
            # pos_1 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)
            # pos_2 = torch.clamp(self.transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget, 0, 1)
            pos_1 = torch.clamp(self.transform(img), 0, 1)
            pos_2 = torch.clamp(self.transform(img), 0, 1)

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

        noise_255 = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
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


# class TransferFloatCIFAR10Pair(CIFAR10):
#     """CIFAR10 Dataset.
#     """
#     def __init__(self, root='data', train=True, transform=None, download=True, perturb_tensor_filepath=None, random_noise_class_path=None, perturbation_budget=1.0):
#         super(TransferCIFAR10Pair, self).__init__(root=root, train=train, download=download, transform=transform)

#         sampled_filepath = os.path.join(root, "sampled_cifar10", "cifar10_1024_4class.pkl")
#         with open(sampled_filepath, "rb") as f:
#             sampled_data = pickle.load(f)
#         if train:
#             self.data = sampled_data["train_data"]
#             self.targets = sampled_data["train_targets"]
#         else:
#             self.data = sampled_data["test_data"]
#             self.targets = sampled_data["test_targets"]

#         if perturb_tensor_filepath != None:
#             self.perturb_tensor = torch.load(perturb_tensor_filepath)
#             self.noise_255 = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
#         else:
#             self.perturb_tensor = None

#         if random_noise_class_path != None:
#             self.random_noise_class = np.load(random_noise_class_path)
#         else:
#             self.random_noise_class = None
        
#         self.perturbation_budget = perturbation_budget

#     # random_noise_class = np.load('noise_class_label.npy')
#         # self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
#         # self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
#         # self.data = self.data.astype(np.float32)
#         # for idx in range(len(self.data)):
#         #     noise = self.noise_255[self.targets[idx]]
#         #     noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location='center')
#         #     self.data[idx] = self.data[idx] + noise
#         #     self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
#         # self.data = self.data.astype(np.uint8)


#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         # print(img[0][0])
#         img = Image.fromarray(img)
#         # print("np.shape(img)", np.shape(img))

#         if self.transform is not None:
#             # print(self.perturb_tensor[self.random_noise_class[index]][0][0])
#             # print("self.transform(img)", self.transform(img).shape)
#             pos_1 = torch.clamp(self.transform((ToTensor_transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget).to('cpu').numpy()), 0, 1)
#             pos_2 = torch.clamp(self.transform((ToTensor_transform(img) + self.perturb_tensor[self.random_noise_class[index]] * self.perturbation_budget).to('cpu').numpy()), 0, 1)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return pos_1, pos_2, target

#     def replace_random_noise_class(self, random_noise_class):
#         # print('length of targets is ', len(self.targets))
#         # print(random_noise_class.shape)
#         # for i in range(10):
#         #     print(i, np.sum(random_noise_class == i))
#         if len(self.targets) == random_noise_class.shape[0]:
#             for i in range(len(self.targets)):
#                 # print(self.targets[i], random_noise_class[i])
#                 self.targets[i] = random_noise_class[i]
#         else:
#             raise('Replacing data noise class failed. Because the length is not consistent.')
    
#     def make_unlearnable(self, random_noise_class, noise):

#         noise_255 = noise.mul(255).clamp_(-255, 255).permute(0, 2, 3, 1).to('cpu').numpy()
#         self.data = self.data.astype(np.float32)
#         if len(self.data) == random_noise_class.shape[0]:
#             for i in range(len(self.targets)):
#                 # print("data:", self.data[i][0])
#                 # print("noise:", noise_255[random_noise_class[i]][0])
#                 self.data[i] += noise_255[random_noise_class[i]]
#                 # input()
#             print("Making data unlearnable done")
#         else:
#             raise('Making data unlearnable failed. Because the length is not consistent.')

#         self.data = self.data.astype(np.uint8)
    
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