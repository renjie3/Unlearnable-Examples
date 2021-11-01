import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_prename = "./results/unlearnable_20211031144658_0.5_512_1000_statistics"
# 20211031144658
pd_reader = pd.read_csv(file_prename+".csv")
# print(pd_reader)

epoch = pd_reader.values[:,0]
loss = pd_reader.values[:,1]
acc = pd_reader.values[:,2]

file_prename = "./results/unlearnable_20211031145034_0.5_512_1000_statistics"
pd_reader = pd.read_csv(file_prename+".csv")
loss1 = pd_reader.values[:,1]
acc1 = pd_reader.values[:,2]

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

p2 = ax.plot(epoch, loss,'r-', label = 'loss')
p2 = ax.plot(epoch, loss1,'g-', label = 'loss1')
ax.legend()
p3 = ax1.plot(epoch,acc, 'b-', label = 'test_acc')
p3 = ax1.plot(epoch,acc1, 'y-', label = 'test_acc1')
ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax1.set_ylabel('acc')
plt.title('Compare loss for different models in training')
plt.savefig(file_prename + ".png")
