import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.style as style
import scienceplots

plt.style.use('science')
# mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'normal'

def extract_accuracy(data_path):
    with open(data_path, 'r') as file:
        lines = file.readlines()

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

rounds = np.array(range(1, 101))

plt.figure(figsize=(8, 6))
plt.plot(rounds, accuracy_plain_fl / 100, linestyle='-', label='Plain FL', linewidth=1.5)
plt.plot(rounds, accuracy_our_protocol / 100, linestyle='--', label='Our Protocol', linewidth=1.5)

ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Helvetica')
    label.set_fontsize(12)
    label.set_fontweight('heavy')

plt.xlabel('Round')
plt.ylabel('Model Accuracy')

plt.legend(loc='lower right', frameon=True, edgecolor='black', framealpha=0.5)

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

plt.xticks(np.arange(0, 101, 10))

plt.savefig('accuracy_comparison_cifar.png')

plt.show()
