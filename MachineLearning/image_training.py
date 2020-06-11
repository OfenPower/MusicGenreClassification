##import and plot cifar10 dataset
import matplotlib.pyplot as plt  #plot lib
import numpy as np
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras


#---load data
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sh = np.shape(x_train)

print("Shape:")
print("- Training-set:\t\t{}".format(np.shape(x_train)))
print("- Training-set:\t\t{}".format(np.shape(y_train)))
print("- Test-set:\t\t{}".format(np.shape(x_test)))
print("- Test-set:\t\t{}".format(np.shape(y_test)))


# --- setup simple cnn with keras for cifar10
model = keras.Sequential([
    keras.layers.Conv2D(128,
                        3,
                        padding='same',
                        activation='relu',
                        input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    #keras.layers.Dropout(p=0.5),
    keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    #keras.layers.Dropout(p=0.5)
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') 
])

# Modelarchitektur anzeigen
model.summary()

#---define hyperparameter---
learning_rate = 0.001
batch_size = 40
epochs = 2
validation_ratio = 0.2  #proportion of training data used for validation



#--- init tensorboard log functionality
logdir = os.path.join("logs",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir,update_freq= 'batch', histogram_freq=1)



#--- train model
sgd = keras.optimizers.SGD(learning_rate=learning_rate)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_train_onehot = keras.utils.to_categorical(y_train)
print(y_train_onehot)
print(np.shape(y_train_onehot))

model.fit(x_train,
          y_train_onehot,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_ratio,
          callbacks=[tensorboard_callback])



# plot confusion matrix

classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

y_pred_onehot=model.predict(x_test)
y_pred=np.argmax(y_pred_onehot, axis=1)


def plot_confusion_matrix(y_true, y_pred, classes):
    cmap=plt.cm.Blues

    # Compute confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred)

    print(conf_matrix)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(conf_matrix.shape[1]),
        yticks=np.arange(conf_matrix.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix',
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt ='d'
    thresh = np.max(conf_matrix )/ 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j,
                    i,
                    format(conf_matrix[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plot_confusion_matrix(y_test, y_pred,classes)
plt.show()