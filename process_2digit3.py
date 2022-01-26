import sys
import numpy as np

group1 = ["0247", "2489", "0568", "3569", "0789", "1248", "3459", "2789", "2456", "2679", "1239", "5679", "1359", "2578", "0389", "1578", "1237", "4789", "1279", "0478", "1348", "0239", "0238", "1269", "2368", "1357", "3568", "2345", "1467", "0157"]
group2 = ["1369", "0367", "1279", "1278", "1356", "3579", "0126", "1356", "0189", "0148", "0467", "1234", "0468", "1346", "2467", "0469", "4568", "0126", "0568", "1239", "0256", "1478", "1567", "0348", "0145", "4689", "0124", "0179", "0389", "4689"]


line1 = sys.stdin.readline()
line2 = sys.stdin.readline()
i = 1
last_data_name = ''
results = []
batch_acc = []
inst_acc = []
while line1 != '':
    number_idx = i // 24
    if i % 12 == 1 and i != 1:
        batch_acc.append(np.average(batch_acc))
        inst_acc.append(np.average(inst_acc))
        # print(batch_acc)
        # print(inst_acc)
        if i % 24 != 1:
            head_str = "\tpredict{}\t\t\t\tpredict{}".format(batch_number, inst_number)
            group1_str = "batch shared:{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(batch_number, *batch_acc, *inst_acc)
        else:
            group2_str = "batch shared:{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(batch_number, *batch_acc, *inst_acc)
            print(head_str)
            # print(group1_str)
            print(group2_str)
        batch_acc = []
        inst_acc = []
        # print('here')
    
    # batch_number = group1[number_idx]
    # inst_number = group2[number_idx]
    line1 = line1.strip('\n')
    line2 = line2.strip('\n')

    if "batch" in line1:
        batch_number = line1.split('test')[1]
        batch_acc.append(float(line2))
    elif "inst" in line1:
        inst_number = line1.split('test')[1]
        inst_acc.append(float(line2))
    # print(line)

    line1 = sys.stdin.readline()
    line2 = sys.stdin.readline()
    i += 2
    