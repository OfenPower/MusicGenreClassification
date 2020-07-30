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
import file_processor

# ------------------------------------------
# ------------------------------------------
# !!! Für Informationen, siehe README.me !!!
# ------------------------------------------
# ------------------------------------------

## ---------------- Klassifikation ----------------

# Pfad zur .json Datei
JSON_PATH = "../data_adjusted_n10.json"

# bool-Flag ob die .json Datei den bereinigten Datensatz enthält oder nicht
is_data_adjusted = True

# Daten aus json laden, in train/validation/test-Split einteilen und für cnn aufbereiten
inputs_train, inputs_test, targets_train, targets_test = file_processor.prepare_cnn_datasets(JSON_PATH, test_size=0.25, is_data_adjusted=is_data_adjusted)

# CNN Modell von Valerio Velardo aufsetzen
cnn_input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
model = keras.models.Sequential([
    # 1. Input Conv Layer = (anzahl mfcc vektoren, n mfcc Koeffizienten, 1) 
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=cnn_input_shape),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),
    
    # 2. Hidden Conv Layer
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=cnn_input_shape),
    keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    keras.layers.BatchNormalization(),

    # 3. Hidden Conv Layer
    keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=cnn_input_shape),
    keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),    
    keras.layers.BatchNormalization(),

    # 4. Flatten und Dense Layer
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),

    # 5. Output Layer
    keras.layers.Dense(10, activation='softmax') 
])

model.summary()


## ---------- Training ----------

# Hyperparameter definieren
learning_rate = 0.0001
batch_size = 32
epochs = 30
validation_ratio = 0.2  # proportion of training data used for validation

# SGD Optimierer festlegen und Model compilen
#sgd = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.5, nesterov=False)
sgd = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# targets in onehot format konvertieren
#targets_train_onehot = keras.utils.to_categorical(targets_train)

# In summary, acc is the accuracy of a batch of training data and val_acc is the accuracy of a batch of testing data.
history = model.fit(inputs_train,
          targets_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=validation_ratio
          )

## Validierung

test_error, test_accuracy = model.evaluate(inputs_test, targets_test, verbose=1)
print("Test Data Accuracy = {}".format(test_accuracy))
#print("Test Data Error = {}".format(test_error))


## Konfusionsmatrix plotten

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


## Traininsverlauf plotten

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

    fig.tight_layout(pad=3.0)

    plt.show()

plot_history(history)