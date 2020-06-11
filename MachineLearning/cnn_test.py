import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt 

print(tf.__version__)

# --- load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sh = np.shape(x_train)
x_train_for_cnn = x_train.reshape([sh[0], sh[1], sh[2],
                                   1])  #reshape to fit as cnn_model input
print("Shape:")
print("- Training-set:\t\t{}".format(np.shape(x_train)))
print("- Training-set:\t\t{}".format(np.shape(y_train)))

# --- setup simple cnn with keras


def create_cnn_model():
    img_inputs = keras.Input(shape=[28, 28, 1])
    conv_1 = keras.layers.Conv2D(16, (4, 4),
                                 activation='relu',
                                 padding='valid',
                                 name='Conv1')(img_inputs)
    conv_2 = keras.layers.Conv2D(16, (5, 5),
                                 activation='relu',
                                 padding='valid',
                                 name='Conv2')(conv_1)
    # convolutional layer
    flatten = keras.layers.Flatten()(conv_2)
    dense_1 = keras.layers.Dense(10, activation='relu')(flatten)
    output = keras.layers.Dense(10, activation='softmax')(dense_1)
    model = keras.models.Model(inputs=img_inputs, outputs=output)
    return model


cnn_model = create_cnn_model()
cnn_model.summary()

# create models for intermediate results -> feature maps
conv_1 = [layer.output for layer in cnn_model.layers if layer.name == 'Conv1']

conv_2 = [layer.output for layer in cnn_model.layers if layer.name == 'Conv2']

feat_map1_model = tf.keras.models.Model(cnn_model.inputs, outputs=conv_1)
feat_map2_model = tf.keras.models.Model(cnn_model.inputs, outputs=conv_2)

#--- train modelX
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.fit(x_train_for_cnn, y_train, epochs=1)


#get feature maps for a specific input image

im_ind=4 #index of the image
image=np.reshape(x_train[im_ind],[1,28,28,1])
feature_map1=feat_map1_model.predict(image)
image=np.reshape(x_train[im_ind],[1,28,28,1])
feature_map2=feat_map2_model.predict(image)


# --- plot
plt.figure(figsize=(5,5))
plt.rcParams.update({'font.size': 14}) #adjust font size
plt.title('Input image')
plt.figure(1,figsize=(16,16))
plt.imshow(x_train[im_ind], cmap='binary')

fig1=plt.figure(figsize=(14,10))
fig1.suptitle('Feature Maps Layer 1')
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.grid(False)
    plt.imshow(np.squeeze(feature_map1[:,:,:,i]), cmap=plt.cm.binary)


fig2=plt.figure(figsize=(14,10))
fig2.suptitle('Feature Maps Layer 2')
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.grid(False)
    plt.imshow(np.squeeze(feature_map2[:,:,:,i]), cmap=plt.cm.binary)

plt.show()