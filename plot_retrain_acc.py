import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# file_prename = "./results/unlearnable_20211025164521_0.5_512_151_budget32.0_class4_retrain_model_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# print(pd_reader)

x_value = np.array([8, 16, 32])
x_value2 = np.array([8])
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]
acc_name = ['nosuffle_bilevel_samplewise', 'suffle_bilevel_samplewise', 'without_training', 'no suffle simclr', 'nosuffle_bilevel_samplewise_eot', ]
color = ['r-', 'g-', 'b-', 'y-', 'c-', 'm-', 'tab:orange', 'tab:purple']
# file_prename_base = "./results/differentiable_20211102231654_0.5_200_512_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:,1]
# acc1 = pd_reader.values[:,2]

acc = []

acc.append(np.array([71.775, 68.85, 59.25]))
acc.append(np.array([78.65, 78.6, 78.5]))
acc.append(np.array([44.02 for _ in range(3)]))
acc.append(np.array([77.95 for _ in range(3)]))
acc.append(np.array([63.6]))

fig, ax=plt.subplots(1,1,figsize=(9,6))
# ax1 = ax.twinx()

for i in range(4):
    _ = ax.plot(x_value, acc[i], color[i], label = acc_name[i])
_ = ax.plot(x_value2, acc[4], 'o', label = acc_name[4])
# p3 = ax.plot(x_value2, acc2, color[1], label = acc_name[1])

ax.legend(loc='upper right')
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
plt.savefig("./visualization/samplewise_retrain_acc_compare.png")

