import numpy as np
import pickle
import os
from sklearn import metrics
import torch

# data_name = "hierarchical64_16_period_dim30_shuffle_diffmean_test1_knn64"
# data_name = "hierarchical64_16_period_dim30_shuffle_diffmean_test2_knn16"
# data_name = "hierarchical32_16_period_dim30_shuffle_diffmean_test1_knn32"
# data_name = "hierarchical32_16_period_dim30_shuffle_diffmean_test2_knn16"
# data_name = "hierarchical32_16_period_dim30_shuffle_diffmean_knn16"
# data_name = "hierarchical16_4_period_dim30_shuffle_diffmean_test1_knn16"
# data_name = "hierarchical16_4_period_dim30_shuffle_diffmean_test2_knn4"
# data_name = "hierarchical32_16_period_dim30_shuffle_std0.03#0.1_diffmean_knn32"
data_name = "hierarchical64_16_period_dim30_shuffle_std0.03#0.1_diffmean_knn64"
sampled_filepath = os.path.join("data", "theory_data", "{}.pkl".format(data_name))
with open(sampled_filepath, "rb") as f:
    sampled_data = pickle.load(f)

print(sampled_data.keys())

train_data = sampled_data["train_data"]
train_targets = sampled_data["train_targets"]

print(metrics.davies_bouldin_score(train_data[:,:,0,0], train_targets))

sample = torch.tensor(train_data[:,:,0,0])
inst_label = torch.tensor(train_targets).cuda()
class_center = []
sort_class = []
intra_class_dis = []
c = np.max(train_targets) + 1
for i in range(c):
    idx_i = torch.where(inst_label == i)[0]
    class_i = sample[idx_i, :]
    class_i_center = class_i.mean(dim=0)
    # print(class_i_center.shape)
    # print((class_i-class_i_center).shape)
    class_center.append(class_i_center)
    intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
    # print(intra_class_dis)
    # input(intra_class_dis[0].shape)
class_center = torch.stack(class_center, dim=0)

class_dis = torch.cdist(class_center, class_center, p=2)

mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

# print(class_dis)
# input()

# print(torch.tensor(intra_class_dis).unsqueeze(1).shape)
intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
# print(intra_class_dis.shape)
# print(trans_intra_class_dis.shape)
intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis

intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

# intra_inter_ratio = intra_class_dis_pair_sum / class_dis

# print(intra_class_dis_pair_sum.shape)
# print(class_dis.shape)
# print(intra_class_dis_pair_sum, class_dis)
loss = torch.max(intra_class_dis_pair_sum / class_dis.cuda(), dim=0)[0].mean()
temp = torch.max(intra_class_dis_pair_sum / class_dis.cuda(), dim=0)[0].detach().cpu().numpy()

print(np.mean(temp))
print(loss)
sample_np = sample.detach().cpu().numpy()
inst_label_np = inst_label.detach().cpu().numpy()
print(metrics.davies_bouldin_score(sample_np, inst_label_np))

# print(np.max(train_targets))
c = np.max(train_targets) + 1
class_center = []
sort_class = []
intra_class_dis = []

# index_range = range(20,30)

for i in range(c):
    idx_i = np.where(train_targets == i)[0]
    class_i = train_data[idx_i,:,0,0]
    class_i_center = train_data[idx_i].mean(axis=0)[:,0,0]
    class_center.append(class_i_center)
    sort_class.append(class_i)
    intra_class_dis.append(np.mean(np.sqrt(np.sum((class_i-class_i_center)**2, axis=1))))


class_center = np.stack(class_center, axis=0)
class_sim = class_center @ class_center.transpose()
class_dis = np.zeros(shape=class_sim.shape)

for i in range(class_center.shape[0]):
    for j in range(class_center.shape[0]):
        dis = np.sqrt(np.sum((class_center[i] - class_center[j]) ** 2))
        class_dis[i,j] = dis

# print(class_dis)

DBindex = []
for i in range(class_center.shape[0]):
    index = []
    for j in range(class_center.shape[0]):
        if j != i:
            index.append((intra_class_dis[i] + intra_class_dis[j]) / class_dis[i,j])
    DBindex.append(np.max(index))

# print(DBindex)
DBindex = np.mean(DBindex)

# print(class_center.shape)
# print(DBindex)

