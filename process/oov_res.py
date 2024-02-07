from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from itertools import chain


def bio_to_entities(tags):
    entities = []
    entity = None
    entity_type = None

    for i, tag in enumerate(tags):
        tag_label, tag_type = tag.split('-') if '-' in tag else (tag, None)

        if tag_label == 'B':
            if entity:
                entities.append((entity_type, tuple(entity)))
            entity = [i]
            entity_type = tag_type
        elif tag_label == 'I':
            if entity and tag_type == entity_type:
                entity.append(i)
            else:
                entity = None
                entity_type = None
        else:
            if entity:
                entities.append((entity_type, tuple(entity)))
                entity = None
                entity_type = None

    if entity:
        entities.append((entity_type, tuple(entity)))

    return entities

def calculate_precision_recall_f1(true_entities, predicted_entities):
    true_entities_dict = defaultdict(list)
    predicted_entities_dict = defaultdict(list)

    for entity_type, entity in true_entities:
        true_entities_dict[entity_type].append(entity)

    for entity_type, entity in predicted_entities:
        predicted_entities_dict[entity_type].append(entity)

    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    correct_predictions_dict = defaultdict(int)
    wrong_predictions_dict = defaultdict(int)

    for label in true_entities_dict.keys():
        true_entities_set = set(true_entities_dict[label])
        predicted_entities_set = set(predicted_entities_dict[label])

        common_entities = true_entities_set.intersection(predicted_entities_set)

        precision = len(common_entities) / (len(predicted_entities_set) + 1e-9)
        recall = len(common_entities) / (len(true_entities_set) + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        precision_dict[label] = precision
        recall_dict[label] = recall
        f1_dict[label] = f1

        # 统计每个标签的预测正确和错误数量
        correct_predictions_dict[label] = len(common_entities)
        wrong_predictions_dict[label] = len(predicted_entities_set - true_entities_set)

    return precision_dict, recall_dict, f1_dict, correct_predictions_dict, wrong_predictions_dict
def find_all_indices(text, text_list):
    indices = []

    for word in text_list:
        start = 0
        while True:
            # 搜索当前词在文本中的位置
            index = text.find(word, start)
            if index == -1:
                break
            indices.append((index))
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
        text_sequence.append(parts[0])  # 读取第二列内容
        true_labels.append(parts[1])    # 读取第三列内容
        predict_labels.append(parts[2])    # 读取第三列内容

text = ''.join(text_sequence)
word_list=string_list_no_spaces


found_indices = find_all_indices(text, word_list)
unique_list = list(set(found_indices))
print(unique_list)
# if found_indices:
#     print("所有重复字符的位置索引：", found_indices)
# else:
#     print("未找到重复字符。")



# 将标签转换为实体级别
true_entities = bio_to_entities(true_labels)
predicted_entities = bio_to_entities(predict_labels)
print(len(true_entities))

true_entities2 = [elem for elem in true_entities if not any(index in elem[1] for index in unique_list)]
predicted_entities2 = [elem for elem in predicted_entities if not any(index in elem[1] for index in unique_list)]
# 计算每个标签的精度、召回率和 F1 值以及每个标签的预测正确和错误数量
precision_dict, recall_dict, f1_dict, correct_predictions_dict, wrong_predictions_dict = calculate_precision_recall_f1(true_entities2, predicted_entities2)

# # 打印每个标签的指标值
# for label in precision_dict.keys():
#     print(f"Label: {label}")
#     print(f"Precision: {precision_dict[label]:.4f}")
#     print(f"Recall: {recall_dict[label]:.4f}")
#     print(f"F1 Score: {f1_dict[label]:.4f}")
#     print(f"Correct Predictions for {label}: {correct_predictions_dict[label]}")
#     print(f"Wrong Predictions for {label}: {wrong_predictions_dict[label]}")
#     print("------------------")
# # 假设这些指标已经计算好并存储在相关变量中

# 打开一个文件用于写入
with open('metrics_report.txt', 'w') as file:
    for label in precision_dict.keys():
        file.write(f"Label: {label}\n")
        file.write(f"Precision: {precision_dict[label]:.4f}\n")
        file.write(f"Recall: {recall_dict[label]:.4f}\n")
        file.write(f"F1 Score: {f1_dict[label]:.4f}\n")
        file.write(f"Correct Predictions for {label}: {correct_predictions_dict[label]}\n")
        file.write(f"Wrong Predictions for {label}: {wrong_predictions_dict[label]}\n")
        file.write("------------------\n")
