# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: data_stats.py
# @time: 2023/8/16 22:25
import matplotlib.pyplot as plt
import json
from operator import itemgetter
from collections import defaultdict

# data stats
cnt_dict = defaultdict(int)
with open('spo.json', 'r') as f:
    content = json.loads(f.read())

for text, triples in content.items():
    cnt_dict[len(triples)] += 1

sorted_cnt_dict = sorted(cnt_dict.items(), key=itemgetter(0))

# bar plot
x_list = [_[0] for _ in sorted_cnt_dict]
y_list = [_[1] for _ in sorted_cnt_dict]

plt.bar(x_list, y_list, color=['r', 'g', 'b'])
plt.xlabel('number of triples')
plt.ylabel('number of samples')
for a, b in zip(x_list, y_list):
    plt.text(a, b, b, ha='center', va='bottom')

plt.savefig('triples_distribution.png')
