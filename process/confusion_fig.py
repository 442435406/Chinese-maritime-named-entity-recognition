
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
# # 混淆矩阵数据
# conf_matrix_data = np.array([[82, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 74, 0, 0, 0, 0, 0, 0],
#                              [0, 4, 276, 0, 0, 0, 0, 0],
#                              [0, 0, 0, 270, 0,0, 1, 3],
#                              [0, 0, 0, 0, 785,13, 0, 0],
#                              [0, 0, 0, 0, 6,360, 32, 0],
#                              [0, 0, 0, 0, 0, 0, 775, 0],
#                              [0, 0, 0, 0, 0, 0, 3, 163]])

# # 标签列表
# labels = ['AT', 'CT', 'CAS', 'DAM', 'LOC', 'ORG', 'SN', 'ST']
# 混淆矩阵数据
conf_matrix_data = np.array([[565, 7, 10],
                             [9, 2266, 87],
                             [19, 83, 4107]])

# 标签列表
labels = ['B', 'I', 'O']

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6.5))
heatmap=sns.heatmap(conf_matrix_data, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels, norm=mcolors.PowerNorm(gamma=0.5),annot_kws={"fontsize": 16})
plt.xlabel('Predicted Labels', fontsize=20)
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 16)

heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 16)
plt.ylabel('True Labels', fontsize=20)
# plt.title('Label Confusion Matrix', fontsize=20)
plt.savefig('E:/code/海上搜救NER/模型/MSAR_NER/image/result_image/confusion_bio.png', dpi=330)
plt.show()

# # 混淆矩阵数据
# conf_matrix_data = np.array([[565, 7, 10],
#                              [9, 2266, 87],
#                              [19, 83, 4107]])

# # 标签列表
# labels = ['B', 'I', 'O']

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # 混淆矩阵数据
# conf_matrix_data = np.array([[82, 0, 0, 0, 0, 0, 0, 0, 0],
#                              [0, 74, 0, 0, 0, 1, 0, 0, 0],
#                              [0, 4, 276, 0, 0, 34, 0, 0, 0],
#                              [0, 0, 0, 270, 0, 17, 0, 1, 3],
#                              [0, 0, 0, 0, 785, 28, 13, 0, 0],
#                              [2, 0, 22, 3, 26, 4107, 0, 44, 5],
#                              [0, 0, 0, 0, 6, 9, 360, 32, 0],
#                              [0, 0, 0, 0, 0, 0, 0, 775, 0],
#                              [0, 0, 0, 0, 0, 8, 0, 3, 163]])

# # 标签列表
# labels = ['AccidentType', 'Cargo', 'Casualty', 'Damage', 'Location', 'O', 'Organization', 'Ship', 'ShipType']

# # 绘制混淆矩阵图
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix_data, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Label Confusion Matrix')
# plt.show()
