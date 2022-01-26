import sys

line = sys.stdin.readline()
last_data_name = ''
while line != '':
    line = line.strip('\n')
    if "python3" in line and "mnist_train_2digit" in line:
        data_name = line.split('--gray_train ')[1].split('--gray_test')[0]
        if last_data_name != data_name:
            print(data_name)
        last_data_name = data_name
    if "unlearnable" in line:
        print(line)
    # print(line)

    line = sys.stdin.readline()
    