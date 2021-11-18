import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# file_prename = "./results/unlearnable_20211025164521_0.5_512_151_budget32.0_class4_retrain_model_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# # print(pd_reader)

# epoch = pd_reader.values[:,0]
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]

# file_prename_base = "./results/differentiable_20211102231654_0.5_200_512_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:,1]
# acc1 = pd_reader.values[:,2]

# fig, ax=plt.subplots(1,1,figsize=(9,6))
# ax1 = ax.twinx()

# p2 = ax.plot(epoch, loss,'r-', label = 'generating model loss')
# p2 = ax.plot(epoch, loss1,'g-', label = 'clean')
# ax.legend()
# # p3 = ax1.plot(epoch,acc, 'b-', label = 'test_acc')
# # p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
# ax1.legend()

# #显示图例
# # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# # plt.legend()
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# # ax1.set_ylabel('acc')
# plt.title('Successful unlearnable loss when retrain and clean test acc \n 4 class * 256 pictures in CIFAR10')
# plt.savefig(file_prename + ".png")

acc_name = ['no_suffle_classwise_32', 'no_suffle_classwise_randoms_start_32', 'no_suffle_samplewise_32', 'no_suffle_samplewise_randoms_start_32', 'clean_simclr']
color = ['r-', 'g-', 'b-', 'y-', 'c-', 'm-', 'tab:orange', 'tab:purple']

# file_prename = "./results/unlearnable_cleantrain_104645774_20211115112200_0.5_512_1000_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# # print(pd_reader)

# epoch1 = pd_reader.values[:,0]
epoch = np.array([i for i in range(0,1000)])
# loss = pd_reader.values[:,1]
# # acc = pd_reader.values[:,2]

# file_prename_base = "./results/unlearnable_samplewise_104645799_20211115112213_0.5_512_1000_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:300,1]
# # acc1 = pd_reader.values[:,2]

    # './results/unlearnable_105461910_3_20211116200102_0.5_512_1000_statistics',
    # './results/unlearnable_105471951_1_20211116200332_0.5_512_1000_statistics',
    # './results/unlearnable_samplewise_105461910_1_20211116200103_0.5_512_1000_statistics',
    # './results/unlearnable_samplewise_105461910_2_20211116200103_0.5_512_1000_statistics',

    # './results/unlearnable_105306665_1_20211116114536_0.5_512_1000_statistics',
    # './results/unlearnable_105306665_2_20211116114536_0.5_512_1000_statistics',
    # './results/unlearnable_samplewise_105306786_1_20211116114616_0.5_512_1000_statistics',
    # './results/unlearnable_samplewise_105306786_2_20211116114616_0.5_512_1000_statistics',

loss = []

file_name_list = [
    './results/unlearnable_105461910_3_20211116200102_0.5_512_1000_statistics',
    './results/unlearnable_105471951_1_20211116200332_0.5_512_1000_statistics',
    './results/unlearnable_samplewise_105461910_1_20211116200103_0.5_512_1000_statistics',
    './results/unlearnable_samplewise_105461910_2_20211116200103_0.5_512_1000_statistics',
    './results/unlearnable_cleantrain_105483054_1_20211116165927_0.5_512_1000_statistics',
    ]

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

for i in range(len(acc_name)):
    file_prename_base = file_name_list[i]
    pd_reader = pd.read_csv(file_prename_base+".csv")
    if i == 4:
        loss.append(pd_reader.values[:,7])
    else:
        loss.append(pd_reader.values[:,8])
    _ = ax.plot(epoch, loss[i], color[i], label = acc_name[i], linewidth =1.5)

# p2 = ax.plot(epoch1, loss,'r-', label = 'clean')
# p2 = ax.plot(epoch, loss1,'g-', label = 'samplewise')
ax.legend()
# p3 = ax1.plot(epoch1,acc, 'b-', label = 'test_acc')
# p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
# ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('denominator')
# ax1.set_ylabel('acc')
plt.title('denominator')
plt.savefig("./visualization/denominator.png")

