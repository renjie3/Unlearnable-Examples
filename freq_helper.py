__author__ = 'Haohan Wang'

import numpy as np
from scipy import signal

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    Images_freq_low = np.array(Images_freq_low)
    Images_freq_high = np.array(Images_freq_high)
    Images_freq_low = np.clip(Images_freq_low, 0, 255).astype(np.uint8)
    Images_freq_high = np.clip(Images_freq_high, 0, 255).astype(np.uint8)
    
    return Images_freq_low, Images_freq_high

if __name__ == '__main__':
    import sys
    version = sys.version_info
    import pickle
    import imageio

    import argparse

    parser = argparse.ArgumentParser(description='ClasswiseNoise')
    parser.add_argument('--file', default='', type=str, help='file to check')
    parser.add_argument('--mnist_targets', action='store_true', default=False)
    args = parser.parse_args()
    # cifar10_1024_4class_grayshift_font_singledigit_mnist
    file_path = './data/sampled_cifar10/{}.pkl'.format(args.file)
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    train_images = data['train_data']
    train_labels = data['train_targets']
    test_images = data['test_data']
    test_labels = data['test_targets']
    
    print(train_images.shape)
    # sys.exit()
    
    r_list = [(i+1)*4 for i in range(7)]

    for r in r_list:
        print(r)
        train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(train_images, r)
        test_image_low_4, test_image_high_4 = generateDataWithDifferentFrequencies_3Channel(test_images, r)
        data['train_data'] = train_image_low_4
        data['test_data'] = test_image_low_4
        file_path = './data/sampled_cifar10/freq_{}_low_{}.pkl'.format(args.file, r)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        data['train_data'] = train_image_high_4
        data['test_data'] = test_image_high_4
        file_path = './data/sampled_cifar10/freq_{}_high_{}.pkl'.format(args.file, r)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

    # train_image_low_4, train_image_high_4 = generateDataWithDifferentFrequencies_3Channel(train_images, 4)
    # np.save('./data/CIFAR10/train_data_low_4', train_image_low_4)
    # np.save('./data/CIFAR10/train_data_high_4', train_image_high_4)

    # train_image_low_8, train_image_high_8 = generateDataWithDifferentFrequencies_3Channel(train_images, 8)
    # np.save('./data/CIFAR10/train_data_low_8', train_image_low_8)
    # np.save('./data/CIFAR10/train_data_high_8', train_image_high_8)

    # train_image_low_12, train_image_high_12 = generateDataWithDifferentFrequencies_3Channel(train_images, 12)
    # np.save('./data/CIFAR10/train_data_low_12', train_image_low_12)
    # np.save('./data/CIFAR10/train_data_high_12', train_image_high_12)

    # train_image_low_16, train_image_high_16 = generateDataWithDifferentFrequencies_3Channel(train_images, 16)
    # np.save('./data/CIFAR10/train_data_low_16', train_image_low_16)
    # np.save('./data/CIFAR10/train_data_high_16', train_image_high_16)

    # eval_image_low_4, eval_image_high_4 = generateDataWithDifferentFrequencies_3Channel(eval_images, 4)
    # np.save('./data/CIFAR10/test_data_low_4', eval_image_low_4)
    # np.save('./data/CIFAR10/test_data_high_4', eval_image_high_4)

    # eval_image_low_8, eval_image_high_8 = generateDataWithDifferentFrequencies_3Channel(eval_images, 8)
    # np.save('./data/CIFAR10/test_data_low_8', eval_image_low_8)
    # np.save('./data/CIFAR10/test_data_high_8', eval_image_high_8)

    # eval_image_low_12, eval_image_high_12 = generateDataWithDifferentFrequencies_3Channel(eval_images, 12)
    # np.save('./data/CIFAR10/test_data_low_12', eval_image_low_12)
    # np.save('./data/CIFAR10/test_data_high_12', eval_image_high_12)

    # eval_image_low_16, eval_image_high_16 = generateDataWithDifferentFrequencies_3Channel(eval_images, 16)
    # np.save('./data/CIFAR10/test_data_low_16', eval_image_low_16)
    # np.save('./data/CIFAR10/test_data_high_16', eval_image_high_16)

    # eval_image_low_20, eval_image_high_20 = generateDataWithDifferentFrequencies_3Channel(eval_images, 20)
    # np.save('./data/CIFAR10/test_data_low_20', eval_image_low_20)
    # np.save('./data/CIFAR10/test_data_high_20', eval_image_high_20)

    # eval_image_low_24, eval_image_high_24 = generateDataWithDifferentFrequencies_3Channel(eval_images, 24)
    # np.save('./data/CIFAR10/test_data_low_24', eval_image_low_24)
    # np.save('./data/CIFAR10/test_data_high_24', eval_image_high_24)

    # eval_image_low_28, eval_image_high_28 = generateDataWithDifferentFrequencies_3Channel(eval_images, 28)
    # np.save('./data/CIFAR10/test_data_low_28', eval_image_low_28)
    # np.save('./data/CIFAR10/test_data_high_28', eval_image_high_28)