#from yapf.yapflib.yapf_api import FormatCode  #enables code beautification

import matplotlib
import matplotlib.pyplot as plt  #plot lib
import numpy as np

classes = np.asarray([1, 2, 3, 4, 5])
# example 
logits = np.array([
    1,
    -1,
    2,
    3,
    1,
])

softmax = np.exp(logits) / sum(np.exp(logits))

#suppose class 4 is correct
onehot= np.array([
    0,
    0,
    0,
    1,
    0,
])

#--------plot
fig = plt.figure(figsize=(14, 6))
ax_1 = fig.add_subplot(131)
im1 = ax_1.bar(classes, logits)
ax_1.title.set_text('Logits')
ax_2 = fig.add_subplot(132)
im2 = ax_2.bar(classes, softmax)
ax_2.set_ylim(0, 1.2)
ax_2.title.set_text('Softmax')
ax_3 = fig.add_subplot(133)
im3 = ax_3.bar(classes, onehot)
ax_3.title.set_text('Onehot')
ax_3.set_ylim(0, 1.2)
plt.show()