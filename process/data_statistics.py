import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split


data = np.load("E:/code/海上搜救NER/模型/MSAR_NER/data/dev_data.npz", allow_pickle=True)
print(len(data['words']))
print(data['labels'])


# 实体数目、占比计算
def entity_count(label_lists):
    total = []
    for labels in label_lists:
        for label in labels:
            if 'B-' in label: # B可以代表该实体
                total.append(label[2:]) #去除前面的'B-'
    entity,count = np.unique(total,return_counts=True)
    entity_num_dic =dict(zip(entity,count))

    return entity_num_dic

entity_num_list = []
entity_ratio_list = []

entity_num_dic = entity_count(data['labels'])
# print(entity_num_dic)
entity_type = list(entity_num_dic.keys())
entity_num = list(entity_num_dic.values())
entity_ratio = [round(num/sum(entity_num), 3) for num in entity_num]
entity_num_list.append(entity_num)
entity_ratio_list.append(entity_ratio)

print(entity_type)
print(entity_num_list)
print(entity_ratio_list)


# 实体数目绘图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

plt.figure(figsize=(10,6))
plt.bar(entity_type, entity_num_list[0], width=0.6, color='teal', alpha=0.8) #'slateblue' 'royalblue' 'teal'
plt.xlabel("实体类型",size=12)
plt.ylabel("实体数量",size=12)
plt.title("各类实体在总数据集中的数量",size=14)
plt.show()