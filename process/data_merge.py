import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split



    

    # 当使用 load 打开保存的 .npz 文件时，会返回一个 NpzFile 对象。这是一个dictionary-like 对象，可以查询其数组列表(使用.files 属性)和数组本身。
data = np.load("./process/pre_train_260.npz", allow_pickle=True)
data_self = np.load("./process/pre_self_20_3.npz", allow_pickle=True)

word_list =[] 
label_list =[] 
length_list =[] 




for i in range(0,260):
    word_list.append(data['words'][i])
    label_list.append(data['labels'][i])
    length_list.append(len(data['words'][i]))
for i in range(0,19):
    word_list.append(data_self['words'][i])
    label_list.append(data_self['labels'][i])
    length_list.append(len(data_self['words'][i]))


print(length_list)



np.savez_compressed("./process/pre_train_280.npz", words=np.array(word_list, dtype=object), labels=np.array(label_list, dtype=object))




