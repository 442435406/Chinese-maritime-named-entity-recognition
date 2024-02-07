from collections import defaultdict

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

    return precision_dict, recall_dict, f1_dict

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

# 将标签转换为实体级别
true_entities = bio_to_entities(true_labels)
predicted_entities = bio_to_entities(predict_labels)
print(predicted_entities)
# 计算每个标签的精度、召回率和 F1 值
precision_dict, recall_dict, f1_dict = calculate_precision_recall_f1(true_entities, predicted_entities)

# 打印每个标签的指标值
for label in precision_dict.keys():
    print(f"Label: {label}")
    print(f"Precision: {precision_dict[label]:.4f}")
    print(f"Recall: {recall_dict[label]:.4f}")
    print(f"F1 Score: {f1_dict[label]:.4f}")
    print("------------------")

