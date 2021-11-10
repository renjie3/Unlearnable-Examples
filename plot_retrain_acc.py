import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# file_prename = "./results/unlearnable_20211025164521_0.5_512_151_budget32.0_class4_retrain_model_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# print(pd_reader)

x_value = np.array([1, 2, 4, 8, 16, 32])
x_value2 = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]
acc_name = ['randomassign_supervised_8', 'randomassign_supervised_32', 'supervised_orglabel_32', 'randomassign_bilevel_selfsupervised_16', 'perturb_on_random_initial_model_32', 'random_noise_32', 'without_training', 'clean simclr']
color = ['r-', 'g-', 'b-', 'y-', 'c-', 'm-', 'tab:orange', 'tab:purple']
# file_prename_base = "./results/differentiable_20211102231654_0.5_200_512_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:,1]
# acc1 = pd_reader.values[:,2]

acc = []

acc.append(np.array([85.975, 85.95, 83.5, 74.45, 56.4, 53.05]))
acc.append(np.array([81.375, 72.075, 63.025, 56.975, 55.725, 56.725]))
acc.append(np.array([84.275, 74.575, 59.4, 56.65, 58.225, 57.5]))
acc.append(np.array([85.65, 84.375, 83.325, 79.15, 72.775, 68.575]))
acc.append(np.array([79.45,72,63.225,55.35,52.925,53.6]))
acc.append(np.array([81.775, 76.5, 67.925, 58.675, 55.35, 49.975]))
acc.append(np.array([44.02 for _ in range(6)]))
acc.append(np.array([85.5 for _ in range(6)]))


acc2 = np.array([81.375, 72.075, 63.025, 60.4, 56.975, 56.95, 57.1, 56.825, 55.725, 56.325, 55.225, 56.325, 53.825, 54.9, 54.95, 57.9, 56.725])

fig, ax=plt.subplots(1,1,figsize=(9,6))
# ax1 = ax.twinx()

for i in range(len(acc)):
    _ = ax.plot(x_value, acc[i], color[i], label = acc_name[i])

# p3 = ax.plot(x_value2, acc2, color[1], label = acc_name[1])

ax.legend()
# p3 = ax1.plot(epoch,acc, 'b-', label = 'test_acc')
# p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
# ax1.legend()

# #显示图例
# # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# # plt.legend()
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# # ax1.set_ylabel('acc')
# plt.title('Successful unlearnable loss when retrain and clean test acc \n 4 class * 256 pictures in CIFAR10')
# plt.savefig(file_prename + ".png")

# file_prename = "./results/unlearnable_20211025164521_0.5_512_151_budget8.0_class10_retrain_model_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# # print(pd_reader)

# epoch1 = pd_reader.values[:,0]
# epoch = np.array([i for i in range(1,1001)])
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]

# file_prename_base = "./results/20211008231026_128_0.5_200_512_1000_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:,1]
# acc1 = pd_reader.values[:,2]

# fig, ax=plt.subplots(1,1,figsize=(9,6))
# ax1 = ax.twinx()

# p2 = ax.plot(epoch1, loss,'r-', label = 'generating model')
# # p2 = ax.plot(epoch1, loss1,'g-', label = 'clean')
# ax.legend()
# p3 = ax1.plot(epoch1,acc, 'b-', label = 'test_acc')
# # p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
# # ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('budget: multiply')
ax.set_ylabel('loss')
# ax1.set_ylabel('acc')
plt.title('retrain SimCLR accuracy')
# plt.savefig("./results/retrain_acc_randomassign_supervised_32.png")
plt.savefig("./results/retrain_acc_compare.png")

