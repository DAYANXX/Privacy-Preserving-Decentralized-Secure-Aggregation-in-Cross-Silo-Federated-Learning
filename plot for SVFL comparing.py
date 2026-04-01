import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.style as style
import scienceplots

plt.style.use('science')
# mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.labelweight'] = 'normal'


num_of_clients = [10,20,30,40,50,60,70,80,90,100]
running_time_per_round_Our = [12.68,20.20,30.78,37.63,42.44,51.92,59.82,68.96,76.30,84.89]
# running_time_per_round_SVFL_server = [1.70,3.23,6.28,12.38,24.59,49.00]
running_time_per_round_SVFL_client = [3359.79,3360.14,3387.29,3370.96,3359.90,3405.93,3404.00,3317.09,3425.35,3334.68]



plt.figure(figsize=(8, 6))
plt.plot(num_of_clients, running_time_per_round_Our,linestyle='--', label='Our Protocol', linewidth=1.5)
plt.plot(num_of_clients, running_time_per_round_SVFL_client, label='SVFL', linewidth=1.5)
# plt.plot(num_of_clients, running_time_per_round_SVFL_server, label='Server of SVFL', linewidth=1.5)

ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Helvetica')
    label.set_fontsize(12)
    label.set_fontweight('heavy')

plt.xlabel('Num of Clients')
plt.ylabel('Running Time(ms)')


plt.legend(loc='best', frameon=True, edgecolor='black', framealpha=0.5)


plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# plt.xticks(np.arange(0, 101, 10))

plt.savefig('per_round_time_vs_num_client_with_SVFL.png')

plt.show()
