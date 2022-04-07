import sys
import numpy as np

line = sys.stdin.readline()
last_data_name = ''
print_str1 = []
print_str2 = []
model_str = []
while line != '':
    line = line.strip('\n')
    if "python" in line:
        theory_train_data = line.split("theory_train_data ")[1].split(" ")[0]
        theory_test_data = line.split("theory_test_data ")[1].split(" ")[0]
        gaussian_aug_std = line.split("gaussian_aug_std ")[1].split(" --")[0]
        # print("{} {} {}".format(theory_train_data, theory_test_data, random_drop_feature_num))
        print_str1.append(float(gaussian_aug_std))
        # print_str2.append(random_drop_feature_num.split(' ')[2])
    elif "unlearnable" in line:
        model_str.append(line)

    line = sys.stdin.readline()

idx = np.argsort(print_str1)
print(idx)
print_aug_str = []
print_model_str = []

for i in idx:
    print_aug_str.append(str(print_str1[i]))
    print_model_str.append(model_str[i])

print(' '.join(print_aug_str))
print(" ".join(print_model_str))
# print(' '.join(print_str2))
    