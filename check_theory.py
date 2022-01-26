import numpy as np
import pickle
import os
from sklearn import metrics

# data_name = "hierarchical64_16_period_dim30_shuffle_diffmean_test1_knn64"
# data_name = "hierarchical64_16_period_dim30_shuffle_diffmean_test2_knn16"
data_name = "hierarchical32_16_period_dim30_shuffle_diffmean_test1_knn32"
# data_name = "hierarchical32_16_period_dim30_shuffle_diffmean_test2_knn16"
# data_name = "hierarchical16_4_period_dim30_shuffle_diffmean_test1_knn16"
# data_name = "hierarchical16_4_period_dim30_shuffle_diffmean_test2_knn4"
sampled_filepath = os.path.join("data", "theory_data", "{}.pkl".format(data_name))
with open(sampled_filepath, "rb") as f:
    sampled_data = pickle.load(f)

print(sampled_data.keys())

train_data = sampled_data["train_data"]
train_targets = sampled_data["train_targets"]
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

DBindex = []
for i in range(class_center.shape[0]):
    index = []
    for j in range(class_center.shape[0]):
        if j != i:
            index.append((intra_class_dis[i] + intra_class_dis[j]) / class_dis[i,j])
    DBindex.append(np.max(index))
DBindex = np.mean(DBindex)

print(class_center.shape)
print(DBindex)

print(metrics.davies_bouldin_score(train_data[:,:,0,0], train_targets))

