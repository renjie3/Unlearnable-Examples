import pickle
import numpy as np
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
    
keys = ['train_data', 'train_targets', 'test_data', 'test_targets']

train_data = data['train_data']
if args.mnist_targets:
    train_targets = data['train_targets_mnist']
else:
    train_targets = data['train_targets']
img_path = './test.png'

class_num = np.max(data["train_targets"]) + 1
if args.mnist_targets:
    class_num = 10 
    
for k in range(class_num):
    i = 0
    while i < len(train_data):
        if train_targets[i] == k:
            imageio.imwrite(img_path, train_data[i])
            print(i)
            c = input(train_targets[i])
            if c == 'n':
                break
            elif c == '':
                pass
            else:
                i = int(c)
                
        i += 1
            
            
train_data = data['test_data']
if args.mnist_targets:
    train_targets = data['test_targets_mnist']
else:
    train_targets = data['test_targets']
img_path = './test.png'
print("test_targets:", np.max(train_targets))
print("test_targets:", len(train_targets))

for k in range(class_num):
    i = 0
    while i < len(train_data):
        if train_targets[i] == k:
            imageio.imwrite(img_path, train_data[i])
            print(i)
            c = input(train_targets[i])
            if c == 'n':
                break
            elif c == '':
                pass
            else:
                i = int(c)
        i += 1
        
