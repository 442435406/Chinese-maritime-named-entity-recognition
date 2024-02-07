import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from itertools import chain

def extract_entities(text_list, label_list):
    entities = []
    current_entity = ''
    current_tag = ''

    for text, label in zip(text_list, label_list):
        if label.startswith('B-'):
            if current_entity != '':
                entities.append(current_entity.strip())
            current_entity = text + ' '
        elif label.startswith('I-'):
            current_entity += text + ' '
        else:
            if current_entity != '':
                entities.append(current_entity.strip())
            current_entity = ''

    return entities



dev_data = np.load("E:/code/海上搜救NER/模型/MSAR_NER/data/dev_data.npz", allow_pickle=True)

dev_text_lists=dev_data['words']
dev_label_lists=dev_data['labels']

dev_entities_list=[]
for dev_text_list,dev_label_list in zip(dev_text_lists,dev_label_lists): 

    # 提取实体
    dev_entities = extract_entities(dev_text_list, dev_label_list)
    dev_entities_list.append(dev_entities)



dev_entities_list = list(chain.from_iterable(dev_entities_list))
# 打印所有实体
print("所有的实体：", len(dev_entities_list))

train_data = np.load("E:/code/海上搜救NER/模型/MSAR_NER/data/train_120.npz", allow_pickle=True)

train_text_lists=train_data['words']
train_label_lists=train_data['labels']

train_entities_list=[]
for train_text_list,train_label_list in zip(train_text_lists,train_label_lists): 

    # 提取实体
    train_entities = extract_entities(train_text_list, train_label_list)
    train_entities_list.append(train_entities)


train_entities_list = list(chain.from_iterable(train_entities_list))
# # # 打印所有实体
# print("所有的实体：", train_entities_list)

# 用于存储重复的元素
duplicates = []

# 遍历列表 a 中的元素，判断是否在列表 b 中存在
for element in dev_entities_list:
    if element not in train_entities_list:
        duplicates.append(element)

# # 输出重复的元素
# print("重复的元素：", duplicates)
# 输出重复的元素
print("重复的元素：", duplicates)
