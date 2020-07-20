import os
import numpy as np
import librosa
import math
import json
from sklearn.model_selection import train_test_split



# Songdateiattribute
SAMPLE_RATE = 22050
DURATION = 30   # in sec
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Mfcc und Labels als json speichern
def save_mfcc(dataset_path, 
              json_path, 
              n_mfcc=13, 
              n_fft=2048, 
              hop_length=512, 
              num_segments=1):  # songs in segmente aufteilen. Für jedes Segment werden n_mffcs berechnet. Dadurch wird die TrainingData um den Faktor num_segments erweitert
    # Daten in Dict speichern
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # Samples per Segment für einen Track mittels num_segments berechnen
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    # mittels der hop_length die anzahl an ergebnis mfcc-vektoren pro samples in einem segment berechnen
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # Alle genres loopen
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # nicht im root ordner?
        if dirpath is not dataset_path:
            
            # genre-bezeichnung speichern
            dirpath_components = dirpath.split("/") # genre/blues = ["genre", "blues"]
            genre_name = dirpath_components[-1]
            data["mapping"].append(genre_name)
            print("\nProcessing {}".format(genre_name))

            # alle songs eines genres durchlaufen
            for f in filenames:
                # audiodatei laden
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # mfccs berechnen
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s 
                    finish_sample = start_sample + num_samples_per_segment
                    #print(start_sample)
                    #print(finish_sample)

                    # Mfcc für ein Segment berechnen. Das Segment wird durch das Intervall [start_sample, finish_sample] gegeben
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length) 

                    # Mfcc transpose nehmen und diese zur Liste machen. 
                    # Dadurch erhält man (num_samples_per_segment/hop_length) viele Mfcc Vektoren, welche n_mfcc viele Koeffizienten beinhalten
                    mfcc = mfcc.T
                    #print(mfcc.shape) 
                    #print(expected_num_mfcc_vectors_per_segment)
                    #print(len(mfcc))
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    # json Datei anlegen
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # listen aus json in ndarrays umwandeln
    inputs = np.array(data["mfcc"])
    labels = np.array(data["labels"])
    return inputs, labels

def prepare_cnn_datasets(data_path, test_size, validation_size):
    """
    Lädt Samples aus .json datei in data_path, splittet diese in
    train, validation und test splits und bereitet die splits
    für das cnn auf, indem die Dimensionen der ndarrays für
    das cnn inputshape angepasst werden.
    """

    # Songdateien laden
    X, Y = load_data(data_path)

    # Train/Test Split erzeugen
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # Train/Validation Split erzeugen
    # Die Ergebnisse sind 3D Vekoren -> (num_samples, anzahl mfcc vektoren, n_mfcc koeffizienten)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size) 

    # ndarray Dimension für CNN Inputshape anpassen
    # Das Ergebnis sind 4D Vektoren -> (num_samples, anzahl mfcc vektoren, n_mfcc koeffizienten, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


# Main zum Testen der file_processing Funktionen
if __name__ == "__main__":
    
     # Input und Output Ordner
    DATASET_PATH = "../genres_short"
    JSON_PATH = "../data_short.json"

    """ save_mfcc(DATASET_PATH, 
              JSON_PATH, 
              n_mfcc=13,  
              n_fft=2048, 
              hop_length=512, 
              num_segments=1) """

    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_cnn_datasets(JSON_PATH, 0.25, 0.2)

    print(X_train.shape)
