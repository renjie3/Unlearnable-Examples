import numpy as np
import pickle
import os

easy_50 = [ 980,  808,  290,  182,  588,   94,  127,  985,  261,  939,  296,  820,
          15,  872,  707,  407,  228, 1018,  669,  684,  799,  974,   54,  933,
         464,  878,   76,  184,  806,  863,  163, 1011,  298,  114,  975,  886,
         797,   55,   78,  171,  375,  199,  378,  629,  320,  390,  208,  635,
         881,  654]

file_path = './data/sampled_cifar10/cifar10_1024_4class.pkl'
with open(file_path, "rb") as f:
    entry = pickle.load(f)

sampled_data = entry["train_data"]
new_data = []
for i in range(len(sampled_data)):
    if i in easy_50:
        continue
    else:
        new_data.append(sampled_data[i])
new_data = np.stack(new_data, axis=0)
print(new_data.shape)
entry["train_data"] = new_data

file_path = './data/sampled_cifar10/cifar10_1024_4class_easy50.pkl'
with open(file_path, "wb") as f:
    pickle.dump(entry, f)
