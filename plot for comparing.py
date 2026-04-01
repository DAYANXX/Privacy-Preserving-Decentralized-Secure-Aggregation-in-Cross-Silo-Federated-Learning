import matplotlib.pyplot as plt
import numpy as np

def extract_accuracy(data_path):
    # 打开文件并读取所有行
    with open(data_path, 'r') as file:
        lines = file.readlines()

    # 遍历每一行并提取准确率数据
    accuracies = []
    for line in lines:
        if 'Training accuracy' in line:
            parts = line.split(',')
            test_acc = parts[3].split(':')[-1].strip()
            accuracies.append(test_acc)

    return accuracies

data_path1 = '../plain_cifar.txt'
accuracy_plain_fl = np.array(extract_accuracy(data_path1), dtype=float)

data_path2 = '../secure_cifar.txt'
accuracy_our_protocol = np.array(extract_accuracy(data_path2), dtype=float)

# 模拟数据：训练轮次
rounds = np.array(range(1, 101))  # 例如，100轮

# 创建图表
plt.figure(figsize=(8, 6))
plt.plot(rounds, accuracy_plain_fl / 100, linestyle='-', color='#4B0082', label='Plain FL', linewidth=2)
plt.plot(rounds, accuracy_our_protocol / 100, linestyle='--', color='#FFA500', label='Our Protocol', linewidth=2)

# 添加标题和标签
plt.title('Comparison of Accuracy Between Plain FL and Our Protocol')
plt.xlabel('Round')
plt.ylabel('Accuracy')

# 调整图例位置和风格
plt.legend(loc='upper left')

# 设置网格样式
plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# 调整坐标轴范围和刻度
plt.xticks(np.arange(0, 101, 10))  # 横轴每隔10显示一次刻度
plt.yticks(np.linspace(0, 1, 11))  # 纵轴0到1之间均匀分布11个刻度

# 保存图表
plt.savefig('accuracy_comparison.png')

# 显示图表
plt.show()
