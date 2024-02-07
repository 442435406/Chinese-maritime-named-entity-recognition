
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from itertools import chain

def find_all_indices(text, text_list):
    indices = []

    for word in text_list:
        start = 0
        while True:
            # 搜索当前词在文本中的位置
            index = text.find(word, start)
            if index == -1:
                break
            indices.append((index,len(word)))
            start = index + 1  # 继续搜索下一个位置

    return indices


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



train_data = np.load("E:/code/海上搜救NER/模型/MSAR_NER/data/train_120.npz", allow_pickle=True)

train_text_lists=train_data['words']
train_label_lists=train_data['labels']

train_entities_list=[]
for train_text_list,train_label_list in zip(train_text_lists,train_label_lists): 

    # 提取实体
    train_entities = extract_entities(train_text_list, train_label_list)
    train_entities_list.append(train_entities)


train_entities_list = list(chain.from_iterable(train_entities_list))

# 用于存储重复的元素
duplicates = []

# 遍历列表 a 中的元素，判断是否在列表 b 中存在
for element in dev_entities_list:
    if element in train_entities_list:
        duplicates.append(element)

string_list_no_spaces = [string.replace(' ', '') for string in duplicates]



file_path = 'E:\\code\\海上搜救NER\\模型\\MSAR_NER\\model\\data.txt'  # 文件路径，替换为你的文件路径
text_sequence = []
true_labels = []
predict_labels = []

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()
    if len(parts) >= 3:
        text_sequence.append(parts[0])  # 读取第一列内容
        true_labels.append(parts[1])    # 读取第二列内容
        predict_labels.append(parts[2])    # 读取第三列内容




word_list=string_list_no_spaces
text = ''.join(text_sequence)

print(text)
print(len(word_list))

found_indices = find_all_indices(text, word_list)
unique_list = list(set(found_indices))

if unique_list:
    print("所有重复字符的位置索引：", len(unique_list))
else:
    print("未找到重复字符。")
