import matplotlib
import matplotlib.pyplot as plt  #plot lib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#  Level | Level for Humans | Level Description                  
# -------|------------------|------------------------------------ 
#  0     | DEBUG            | [Default] Print all messages       
#  1     | INFO             | Filter out INFO messages           
#  2     | WARNING          | Filter out INFO & WARNING messages 
#  3     | ERROR            | Filter out all messages      

import tensorflow as tf
import numpy as np

image = np.eye(12) 
image[8 - 1:12 - 1, 0:4] = np.eye(4)
image[0:4, 7:11] = np.eye(4)

#reshape for to apply tf conv function
image_ten = image.reshape([1, 12, 12, 1])

kernel = tf.eye(4)
kernel_ten = tf.reshape(kernel, [4, 4, 1, 1])

conv_result = tf.nn.conv2d(image_ten,
                           kernel_ten,
                           strides=[1, 1],
                           padding="VALID")
out_image = tf.squeeze(conv_result)

#--------plot
fig = plt.figure(figsize=(14, 6))
ax_1 = fig.add_subplot(131)
im1 = ax_1.imshow(image)
ax_1.title.set_text('Eingabebild')
fig.colorbar(im1, ax=ax_1)
ax_2 = fig.add_subplot(132)
im2 = ax_2.imshow(kernel)
ax_2.title.set_text('Filter/Faltungskern')
fig.colorbar(im2, ax=ax_2)
ax_3 = fig.add_subplot(133)
im3 = ax_3.imshow(out_image)
ax_3.title.set_text('Resultat')
fig.colorbar(im3, ax=ax_3)
plt.show()