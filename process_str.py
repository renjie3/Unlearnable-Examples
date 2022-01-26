import sys

line = sys.stdin.readline()
last_data_name = ''
print_line_str = []
while line != '':
    line = line.strip('\n')
    print_line_str.append(line)
        
    # print(line)

    line = sys.stdin.readline()
    
print(' '.join(print_line_str))