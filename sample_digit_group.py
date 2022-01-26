import numpy as np
import pickle
import os
from itertools import combinations
import random

import argparse

parser = argparse.ArgumentParser(description='ClasswiseNoise')
args = parser.parse_args()

comb_factory = list(combinations(range(10), 4))
comb_factory = random.sample(comb_factory, 30)
group1_str = []
group2_str = []

for comb in comb_factory:
    group2 = []
    for i in range(10):
        if i not in comb:
            group2.append(i)
    group2 = list(combinations(group2, 4))
    group2 = random.sample(group2, 1)[0]
    group1_str.append("{}{}{}{}".format(*comb))
    group2_str.append("{}{}{}{}".format(*group2))

group1_str = ' '.join(group1_str)
group2_str = ' '.join(group2_str)

print(group1_str)
print(group2_str)