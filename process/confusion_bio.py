from collections import defaultdict

def calculate_confusion_matrix(true_labels, predicted_labels):
    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for true, pred in zip(true_labels, predicted_labels):
        true_label, true_entity = true.split('-') if '-' in true else (true, 'O')
        pred_label, pred_entity = pred.split('-') if '-' in pred else (pred, 'O')

        confusion_matrix[true_label][pred_label] += 1

    return confusion_matrix

# # 示例真实标签和预测标签
# true_labels = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
# predicted_labels = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'B-ORG', 'I-ORG', 'O', 'O']

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



# 计算混淆矩阵
conf_matrix = calculate_confusion_matrix(true_labels, predict_labels)


# 打印混淆矩阵
print("Confusion Matrix:")
for true_label in conf_matrix:
    print(f"True Label: {true_label}")
    for pred_label, count in conf_matrix[true_label].items():
        print(f"Predicted Label: {pred_label}, Count: {count}")
