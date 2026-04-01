import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.style as style
import scienceplots

plt.style.use('science')
# mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'normal'


num_of_clients = [86.76,164.88,321.13,633.63,1258.63,2508.63]
running_time_per_round = [5.68,7.09,8.77,11.56,33.91,59.12]



plt.figure(figsize=(8, 6))
plt.plot(num_of_clients, running_time_per_round, linestyle='-', label='Per Client', linewidth=1.5)
# plt.plot(rounds, accuracy_our_protocol / 100, linestyle='--', label='Our Protocol', linewidth=1.5)

ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Helvetica')
    label.set_fontsize(12)
    label.set_fontweight('heavy')

plt.xlabel('Model Parameter Vector Size(KB)')
plt.ylabel('Running Time(ms)')

plt.legend(loc='upper left', frameon=True, edgecolor='black', framealpha=0.5)

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# plt.xticks(np.arange(0, 101, 10))
plt.savefig('running_time_per_round_vs_vector_size.png')

plt.show()
