import sys

line = sys.stdin.readline()
last_data_name = ''
test_batch_data_name = []
test_inst_data_name = []
model = []
while line != '':
    line = line.strip('\n')
    if "mnist_train_2digit" in line:
        # data_name.append(line)
        batch_name = line.split('batch')[1].split('_')[0]
        inst_name = line.split('batch')[1].split('_')[1].strip(' ')
        for _ in range(3):
            test_batch_data_name.append("mnist_train_2digit_test{}".format(batch_name))
            test_inst_data_name.append("mnist_train_2digit_test{}".format(inst_name))

    if "unlearnable" in line:
        model.append(line)
    # print(line)

    line = sys.stdin.readline()

print(' '.join(model))
print(' '.join(test_batch_data_name))
print(' '.join(test_inst_data_name))
