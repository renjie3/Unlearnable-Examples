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

acc_name = ['randomassign_supervised_8', 'randomassign_supervised_32', 'supervised_orglabel_32', 'randomassign_bilevel_selfsupervised_16', 'perturb_on_random_initial_model_32', 'random_noise_32', 'clean simclr']
color = ['r-', 'g-', 'b-', 'y-', 'c-', 'm-', 'tab:orange', 'tab:purple']

# file_prename = "./results/unlearnable_20211025164521_0.5_512_151_budget8.0_class4_retrain_model_statistics"
# # 20211031144658
# pd_reader = pd.read_csv(file_prename+".csv")
# # print(pd_reader)

# epoch1 = pd_reader.values[:,0]
epoch = np.array([i for i in range(1,1001)])
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]

# file_prename_base = "./results/differentiable_20211102231654_0.5_200_512_statistics"
# pd_reader = pd.read_csv(file_prename_base+".csv")
# loss1 = pd_reader.values[:,1]
# acc1 = pd_reader.values[:,2]

loss = []

file_name_list = [
    './results/randomassign_supervised_10class_classwise_8_10_0.8_checkpoint_perturbation_budget8.0_class4_retrain_model_statistics',
    './results/randomassign_supervised_10class_classwise_32_10_0.8_checkpoint_perturbation_budget8.0_class4_retrain_model_statistics',
    './results/supervised_orglabel_4class_perturbation_budget8.0_class4_orglabel_retrain_model_statistics',
    './results/unlearnable_20211025164521_0.5_512_151_budget8.0_class4_retrain_model_statistics',
    './results/differentiable_20211102231654_0.5_200_512_32.0_30_0.8_perturb_on_random_initial_model_budget8.0_class4_retrain_model_statistics',
    './results/random_noise32_perturbation_budget8.0_class4_retrain_model_statistics',
    './results/differentiable_20211102231654_0.5_200_512_statistics'
    ]

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

for i in range(len(acc_name)):
    file_prename_base = file_name_list[i]
    pd_reader = pd.read_csv(file_prename_base+".csv")
    loss.append(pd_reader.values[:,1])
    _ = ax.plot(epoch, loss[i], color[i], label = acc_name[i], linewidth =1.5)

# p2 = ax.plot(epoch1, loss,'r-', label = 'generating model')
# p2 = ax.plot(epoch1, loss1,'g-', label = 'clean')
ax.legend()
# p3 = ax1.plot(epoch1,acc, 'b-', label = 'test_acc')
# p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
# ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
# ax1.set_ylabel('acc')
plt.title('Retrain loss vs. clean SimCLR loss')
plt.savefig("./visualization/retrain_loss.png")

