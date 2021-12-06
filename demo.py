import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from numpy.core.arrayprint import format_float_positional
from numpy.lib.index_tricks import AxisConcatenator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import matplotlib.animation as animation
import os
import datetime
import imageio

class_num = 3
data_num = 10
iteration_step = 50
class_norm_center = [[0,2],[-1.73,-1],[1.73,-1]]
# class_norm_center = [[0,1],[0,-1],[1,0],[-1,0]]
# feat_norm_center = [[0,20],[-17.3,-10],[17.3,-10]]
feat_norm_center = np.array([[0,20],[-17.3,-10],[17.3,-10]]) * 0.1
use_delta = False
data_normal_scale = 10
feat_normal_scale = 10
use_anchor_extractor = False
use_feat_extrator = True

# good result
# 20211201162605
# 20211201164016
# 20211201165947
# 20211201184138
# 20211201192839

def get_length(x):
    return np.sqrt(x[0]**2 + x[1]**2)

def normalized(data):
    if len(data.shape) == 2:
        for i in range(class_num * data_num):
            length = np.sqrt(data[i,0]**2 + data[i,1]**2)
            data[i,0] = data[i,0] / length
            data[i,1] = data[i,1] / length
        return data
    elif len(data.shape) == 1:
        length = np.sqrt(data[0]**2 + data[1]**2)
        data[0] = data[0] / length
        data[1] = data[1] / length
        return data

initial_data = []
feature_extractor = []
for i in range(class_num):
    x1 = np.random.normal(loc=class_norm_center[i][0], scale=data_normal_scale, size=10)
    x2 = np.random.normal(loc=class_norm_center[i][1], scale=data_normal_scale, size=10)
    initial_data.append(np.stack((x1,x2), axis=1))
    feat1 = np.random.normal(loc=feat_norm_center[i][0], scale=feat_normal_scale, size=10)
    feat2 = np.random.normal(loc=feat_norm_center[i][1], scale=feat_normal_scale, size=10)
    feature_extractor.append(np.stack((feat1,feat2), axis=1))
input_data = normalized(np.concatenate(initial_data, axis=0))
feature_extractor = np.concatenate(feature_extractor, axis=0)

delta = np.random.rand(class_num * data_num, 2) * 0.1 - 0.05
data_pos_1 = input_data + delta
data_pos_2 = input_data - delta

data_pos_1 = normalized(data_pos_1)
data_pos_2 = normalized(data_pos_2)
gif_images = []
pre_name = './demo/{}'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

if use_delta:
    last_pos_1 = data_pos_1
    last_pos_2 = data_pos_2
    pass
else:
    last_input = input_data
    last_feature = feature_extractor
    for _step in range(iteration_step):
        adjusted_data1 = []
        adjusted_data2 = []
        for i in range(class_num * data_num):
            force_dir = []
            force_size = []
            force = []
            for j in range(class_num * data_num):
                # print(normalized(last_pos_1[i] - last_pos_1[j]))
                if i != j:
                    temp_dir = normalized(last_input[j] - last_input[i])
                    temp_force = (np.exp(np.dot(last_input[i], last_input[j])) - np.exp(get_length(last_input[i] - last_input[j]))) * 0.005
                    force_dir.append(temp_dir)
                    force_size.append(temp_force)
                    force.append(temp_dir * temp_force)
                # print(np.exp(np.dot(last_pos_1[i], last_pos_1[j])))
                # print(np.exp(get_length(last_pos_1[i] - last_pos_1[j])))
                # print(force_size)
            force = np.array(force).mean(axis=0)
            adjusted_data1.append(last_input[i] + force)
            
        # adjusted_data1 = last_input
        
        if use_feat_extrator:
            adjusted_feature = []
            my_alpha = 0.95
            if use_anchor_extractor:
                for i in range(class_num):
                    for j in range(data_num):
                        idx = i*data_num + j
                        class_feat_ave = last_feature[i*data_num:(i+1)*data_num].mean()
                        adjusted_feature.append(last_feature[idx] * my_alpha + (1-my_alpha) * class_feat_ave)
            else:
                for i in range(class_num * data_num):
                    pulled_samples = []
                    pulled_weight = [] # the distance changed or |x-y|
                    pulled_feature = []
                    for j in range(class_num * data_num):
                        if i != j:
                            old_dis = get_length(last_input[j] - last_input[i])
                            new_dis = get_length(adjusted_data1[j] - last_input[i]) # TODO
                            # print(new_dis)
                            if new_dis < old_dis:
                                pulled_samples.append(j)
                                pulled_weight.append(old_dis - new_dis)
                                pulled_feature.append(last_feature[j] * (old_dis - new_dis))
                    if len(pulled_feature) == 0:
                        changed_feat = 0
                    else:
                        changed_feat = np.array(pulled_feature).mean(axis=0)
                        # print(changed_feat.shape)
                    adjusted_feature.append(last_feature[i] + 10 * changed_feat)
                    # print(changed_feat)
                
            # input()
            topk_min_index = []
            topk_min_list = []
            
            topk_max_index = []
            topk_max_list = []
            
            adjusted_feature = np.array(adjusted_feature)
            for i in range(class_num * data_num):
                topk_toberank = []
                feature_toberank = adjusted_feature - adjusted_feature[i]
                for j in range(len(feature_toberank)):
                    topk_toberank.append(get_length(feature_toberank[j]))
            # print(adjusted_feature)
                topk_toberank = np.array(topk_toberank)
                topk_min = topk_toberank[np.argpartition(topk_toberank, 6)[:6]]
                topk_min_index.append(np.argpartition(topk_toberank, 6)[:6])
                topk_min_list.append(topk_min)
                
                topk_max = topk_toberank[np.argpartition(topk_toberank, 5)[:5]]
                topk_max_index.append(np.argpartition(topk_toberank, -5)[-5:])
                topk_max_list.append(topk_max)
            
            for i in range(class_num * data_num):
                sum_neighbor = 0
                sum_weight = 0
                sum_far_neighbor = 0
                sum_far_weight = 0
                for j in range(len(topk_min_index[i])):
                    sum_neighbor += adjusted_data1[topk_min_index[i][j]] * topk_min_list[i][j]
                    sum_weight += topk_min_list[i][j]
                for j in range(len(topk_max_index[i])):
                    sum_far_neighbor += adjusted_data1[topk_max_index[i][j]] * topk_max_list[i][j]
                    sum_far_weight += topk_max_list[i][j]
                # print(sum_neighbor)
                # print(sum_weight)
                avg_neighbor = sum_neighbor / sum_weight
                avg_far_neighbor = sum_far_neighbor / sum_far_weight
                new_pos = adjusted_data1[i] + 0.1 * avg_neighbor - 0.2 * avg_far_neighbor
                adjusted_data2.append(normalized(new_pos))
            
            adjusted_data2 = np.array(adjusted_data2)
        else:
            my_alpha = 0.95
            if use_anchor_extractor:
                for i in range(class_num):
                    for j in range(data_num):
                        idx = i*data_num + j
                        class_feat_ave = last_input[i*data_num:(i+1)*data_num].mean(axis=0)
                        far_feat_ave = 0
                        if i != 0:
                            far_feat_ave += last_input[0:i*data_num].mean(axis=0)
                        if i != class_num-1:
                            far_feat_ave += last_input[(i+1)*data_num:class_num*data_num].mean(axis=0)
                        far_feat_ave *= 0.5
                        # print(class_feat_ave.shape)
                        new_pos = last_input[idx] * my_alpha + (1-my_alpha) * (class_feat_ave - far_feat_ave)
                        adjusted_data2.append(normalized(new_pos))
                        # input()
            adjusted_data2 = np.array(adjusted_data2)
            
        
        color_list = ['r', 'g', 'b', 'y']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(2, 2, 1)
        for i in range(class_num):
            plt.title("step {}".format(_step))
            plt.scatter(last_input[i*data_num:(i+1)*data_num, 0], last_input[i*data_num:(i+1)*data_num, 1], s=21, alpha=0.3, c=color_list[i], cmap=plt.cm.Spectral)
            plt.xlim((-1.2, 1.2))
            plt.ylim((-1.2, 1.2))
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
        for i in range(3):
            ax = fig.add_subplot(2, 2, i+2)
            plt.title("step {}".format(_step))
            plt.scatter(last_input[i*data_num:(i+1)*data_num, 0], last_input[i*data_num:(i+1)*data_num, 1], s=21, alpha=0.3, c=color_list[i], cmap=plt.cm.Spectral)
            plt.xlim((-1.2, 1.2))
            plt.ylim((-1.2, 1.2))
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())

        if not os.path.exists(pre_name):
            os.mkdir(pre_name)
        plt.savefig('{}/step_{}.png'.format(pre_name, _step))
        plt.close()
        
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(class_num):
            plt.title("step {}".format(_step))
            plt.scatter(adjusted_feature[i*data_num:(i+1)*data_num, 0], adjusted_feature[i*data_num:(i+1)*data_num, 1], s=21, alpha=0.3, c=color_list[i], cmap=plt.cm.Spectral)
            plt.xlim((-50, 50))
            plt.ylim((-50, 50))
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
        if not os.path.exists(pre_name):
            os.mkdir(pre_name)
        plt.savefig('{}/feature_step_{}.png'.format(pre_name, _step))
        plt.close()
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        for i in range(class_num):
            plt.title("step {}".format(_step))
            plt.scatter(adjusted_feature[i*data_num:(i+1)*data_num, 0], adjusted_feature[i*data_num:(i+1)*data_num, 1], s=21, alpha=0.3, c=color_list[0], cmap=plt.cm.Spectral)
            plt.xlim((-50, 50))
            plt.ylim((-50, 50))
            ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
            ax.yaxis.set_major_formatter(NullFormatter())
        if not os.path.exists(pre_name):
            os.mkdir(pre_name)
        plt.savefig('{}/feature_1color_step_{}.png'.format(pre_name, _step))
        plt.close()
        
        last_input = adjusted_data2
        last_feature = adjusted_feature


        gif_images.append(imageio.imread('{}/step_{}.png'.format(pre_name, _step)))   # 读取图片
    
    imageio.mimsave('{}/all.gif'.format(pre_name), gif_images, fps=10)   # 转化为gif动画
        