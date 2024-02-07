from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

def calculate_label_confusion_matrix(true_labels, predicted_labels):
    label_confusion_matrix = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(true_labels, predicted_labels):
        true_label = true.split('-')[-1] if '-' in true else true
        pred_label = pred.split('-')[-1] if '-' in pred else pred

        label_confusion_matrix[true_label][pred_label] += 1

    return label_confusion_matrix

def print_confusion_matrix(matrix):
    labels = sorted(matrix.keys())
    print('\t' + '\t'.join(labels))
    for true_label in labels:
        print(true_label, end='\t')
        for pred_label in labels:
            print(matrix[true_label][pred_label], end='\t')
        print()
file_path = 'E:\\code\\海上搜救NER\\模型\\MSAR_NER\\model\\data7.txt'  # 文件路径，替换为你的文件路径
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



# 计算标注类别之间的混淆矩阵
label_conf_matrix = calculate_label_confusion_matrix(true_labels, predict_labels)

# 打印标注类别之间的混淆矩阵
print("Label Confusion Matrix:")
for true_label in label_conf_matrix:
    print(f"True Label: {true_label}")
    for pred_label, count in label_conf_matrix[true_label].items():
        print(f"Predicted Label: {pred_label}, Count: {count}")
# 打印标注类别之间的混淆矩阵（矩阵形式）
print("Label Confusion Matrix:")
print_confusion_matrix(label_conf_matrix)
# # 绘制混淆矩阵图
# plt.figure(figsize=(8, 6))
# sns.heatmap(label_conf_matrix, annot=True, cmap='Blues', fmt='d')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()