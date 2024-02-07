import json
import matplotlib.pyplot as plt

# 从本地JSON文件中读取数据
with open("C:/Users/Administrator/Desktop/bert预训练/models/100m/pretrain_bert_base_100m_32_30/trainer_state.json", 'r') as json_file:
    data = json.load(json_file)

# 提取epoch、loss和learning_rate数据
epochs = [entry["epoch"] for entry in data["log_history"]]
loss_values = [entry["loss"] for entry in data["log_history"]]
learning_rates = [entry["learning_rate"] for entry in data["log_history"]]

# 创建画布和两个坐标轴
fig, ax1 = plt.subplots()

# 绘制左纵轴数据（loss）为曲线
ax1.plot(epochs, loss_values, linestyle='-', color='#00B0F0', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params('y')

# 创建右纵轴坐标轴
ax2 = ax1.twinx()
# 绘制右纵轴数据（learning_rate）为曲线
ax2.plot(epochs, learning_rates, linestyle='--', color='#5B9BD5', label='Learning Rate')
ax2.set_ylabel('Learning Rate')
ax2.tick_params('y')

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# 添加标题
plt.title('Loss and Learning Rate vs. Epoch')

# 显示图表
plt.grid(True)
plt.show()
