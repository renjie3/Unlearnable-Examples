from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

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
            

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
