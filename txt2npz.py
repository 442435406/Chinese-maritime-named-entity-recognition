import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


raw_data = 'self_labeld20_3.txt'
word_list = []
label_list = []

if os.path.exists("pre_self_20_3.npz") is True:
    print('Processed_data Exists')

else:
    row = 0
    with open(raw_data, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        for line in f:
            if len(line) >= 4:
                temp = line.strip('\n').split()
                if len(temp) == 1:
                    temp.insert(0,' ') # 原文本中可能有空格，如果直接split会把原始文本空格给去了，temp列表中就只有一个元素了
                word, label = temp
                words.append(word)
                labels.append(label)
            else:
                if len(words) != 0:
                    row = row + 1
                    word_list.append(words)
                    label_list.append(labels)
                    words = []
                    labels = []

    print('There are {} sentences in total'.format(row))
    np.savez_compressed("pre_self_20_3.npz", words=np.array(word_list, dtype=object), labels=np.array(label_list, dtype=object))
