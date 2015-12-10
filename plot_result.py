import numpy as np
import matplotlib.pyplot as plt

Methods = ['No block', 'Block-wise parallel\nserial within block', 'Block-wise parallel\nparallel within block', 'Coalesced\nblock read/write']
M_ratio = [6.7283578948, 15.6358873444, 11.4581350649, 13.047501137]

fig, ax = plt.subplots()

index = np.arange(len(Methods))
bar_width = 0.35

rects1 = plt.bar(index + bar_width, M_ratio, bar_width,
                 color='b')

plt.xlabel('Methods', fontsize=16)
plt.ylabel('Speed Up over Serial', fontsize=16)
plt.title('Methods Comparison', fontsize=18)
plt.xticks(index + bar_width, Methods, rotation=25, fontsize=14)
#plt.legend()

plt.tight_layout()
plt.show()

Blocksize = ['(8, 8)', '(16, 16)', '(32, 32)', '(16, 64)', '(64, 16)']
B_ratio = [6.92745628371, 10.8556716177, 13.047501137, 8.16667828547, 8.49435139768]

fig, ax = plt.subplots()

index = np.arange(len(Blocksize))
bar_width = 0.35

rects1 = plt.bar(index + bar_width, B_ratio, bar_width,
                 color='r')

plt.xlabel('Sizes', fontsize=16)
plt.ylabel('Speed Up over Serial', fontsize=16)
plt.title('Block Size Comparison', fontsize=18)
plt.xticks(index + bar_width, Blocksize, rotation=25, fontsize=14)
#plt.legend()

plt.tight_layout()
plt.show()
