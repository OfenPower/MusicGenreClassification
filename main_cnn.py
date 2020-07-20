# Libraries
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#  Level | Level for Humans | Level Description                  
# -------|------------------|------------------------------------ 
#  0     | DEBUG            | [Default] Print all messages       
#  1     | INFO             | Filter out INFO messages           
#  2     | WARNING          | Filter out INFO & WARNING messages 
#  3     | ERROR            | Filter out all messages      
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt 
import librosa
import librosa.display
from sklearn.model_selection import train_test_split

# Custom imports
import feature_extract
import file_processing


# Input und Output Ordner
DATASET_PATH = "../genres_short"
JSON_PATH = "../data_short.json"



## ---------------- Vorverarbeitung ----------------



## ---------------- Extraktion von Merkmalen ----------------
""" file_processing.save_mfcc(DATASET_PATH, 
                          JSON_PATH, 
                          n_mfcc=13,  
                          n_fft=2048, 
                          hop_length=512, 
                          num_segments=1) """


## ---------------- Dimensionsreduzierung ----------------



## ---------------- Klassifikation ----------------

# Daten aus json laden, in train/validation/test-Split einteilen und für cnn aufbereiten
inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = file_processing.prepare_cnn_datasets(JSON_PATH, test_size=0.25, validation_size=0.2)

# NN Modell bauen
model = keras.models.Sequential([
    # Flatten Layer als Input Layer nehmen.
    # inputs_train.shape[1] = track_samples / hop_length viele Mfcc Vektoren 
    # inputs_train.shape[2] = n_mfcc Koeffizienten
    keras.layers.Flatten(input_shape=(inputs_train.shape[1], inputs_train.shape[2])),    
    keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax'),
])

model.summary()


## ---------- Training ----------

# Hyperparameter definieren
learning_rate = 0.0001
batch_size = 32 #5
epochs = 50 #10
validation_ratio = 0.2  # proportion of training data used for validation

# SGD Optimierer festlegen und Model compilen
#sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.5, nesterov=False)
sgd = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# targets in onehot format konvertieren
targets_train_onehot = keras.utils.to_categorical(targets_train)
#print(targets_train)
#print(targets_train_onehot)
#print(targets_train.shape)

# In summary, acc is the accuracy of a batch of training data and val_acc is the accuracy of a batch of testing data.
history = model.fit(inputs_train,
          targets_train_onehot,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_ratio
          )

## Validierung

#model.evaluate(inputs_test, targets_test, verbose=1)

classes = [
    'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop',
    'reggae', 'rock'
]

y_pred_onehot = model.predict(inputs_test)
y_pred = np.argmax(y_pred_onehot, axis=1)


def plot_confusion_matrix(y_test, y_pred, classes):
    cmap=plt.cm.Blues

    # Compute confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_test, y_pred)

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


plot_confusion_matrix(targets_test, y_pred,classes)
plt.show()


def plot_history(history):
    fig, ax = plt.subplots(2)

    # Accuracy Plot
    ax[0].plot(history.history["accuracy"], label="train_accuracy")
    ax[0].plot(history.history["val_accuracy"], label="test_accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy eval")

    # Error Plot
    ax[1].plot(history.history["loss"], label="train error")
    ax[1].plot(history.history["val_loss"], label="test error")
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Error eval")

    plt.show()

plot_history(history)