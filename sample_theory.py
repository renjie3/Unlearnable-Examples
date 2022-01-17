import numpy as np
import pickle
import os

mu0 = 1
sigma0 = 1

delta1 = 0
sigma1 = 0.1
q_mu1 = 1
p_mu1 = 1 + delta1

sigma2 = 10
temp = 4*(sigma1**2 * sigma0**2 + sigma1**2 * mu0**2) * (sigma2**2 * sigma0**2 + sigma2**2 * mu0**2)
delta2 = 0
q_mu2 = 1
p_mu2 = 1 + delta2

# print((sigma1**2 * sigma0**2 + sigma1**2 * mu0**2) * 2 / delta1**2)
# print( delta2**2 / ((sigma2**2 * sigma0**2 + sigma2**2 * mu0**2) * 2))

train_size = 10000

X1 = np.random.normal(loc=mu0, scale=sigma0, size=train_size)
X2 = np.random.normal(loc=mu0, scale=sigma0, size=train_size)

Q1 = np.random.normal(loc=q_mu1, scale=sigma1, size=train_size)
P1 = np.random.normal(loc=p_mu1, scale=sigma1, size=train_size)
Q2 = np.random.normal(loc=q_mu2, scale=sigma2, size=train_size)
P2 = np.random.normal(loc=p_mu2, scale=sigma2, size=train_size)

U1 = X1 * Q1
V1 = X1 * P1
U2 = X2 * Q2
V2 = X2 * P2

U = np.stack([U1,U2], axis=1)
V = np.stack([V1,V2], axis=1)
train_data = np.expand_dims(np.stack([U,V], axis=1), axis=3).astype(np.float32)


train_targets = [0 for _ in range(train_size)]


test_size = 5000

X1 = np.random.normal(loc=mu0, scale=sigma0, size=test_size)
X2 = np.random.normal(loc=mu0, scale=sigma0 * 5, size=test_size)

U = np.stack([X1,X2], axis=1)
V = np.stack([X1,X2], axis=1)
test_data_class0 = np.stack([U,V], axis=1)

X1 = np.random.normal(loc=-mu0, scale=sigma0, size=test_size)
X2 = np.random.normal(loc=mu0, scale=sigma0 * 5, size=test_size)

U = np.stack([X1,X2], axis=1)
V = np.stack([X1,X2], axis=1)
test_data_class1 = np.stack([U,V], axis=1)

test_data = np.expand_dims(np.concatenate([test_data_class0, test_data_class1], axis=0), axis=3).astype(np.float32)

test_target_0 = [0 for _ in range(test_size)]
test_target_1 = [1 for _ in range(test_size)]
test_targets = test_target_0 + test_target_1

sampled = {}
sampled["train_data"] = train_data
sampled["train_targets"] = train_targets
sampled["test_data"] = test_data
sampled["test_targets"] = test_targets

file_path = './data/theory_data/data2.pkl'
with open(file_path, "wb") as f:
    entry = pickle.dump(sampled, f)
