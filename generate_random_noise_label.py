import numpy as np

a = np.random.randint(10, size=1024)

print(a)

np.save('noise_class_label_1024_10class.npy', a)