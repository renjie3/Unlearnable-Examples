import numpy as np

a = np.random.randint(4, size=4000)

print(a)

np.save('noise_class_label_1024_4class_test.npy', a)