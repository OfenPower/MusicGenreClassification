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
            dirpath_components = dirpath.split("/") # macht aus "genre/blues" = ["genre", "blues"]
            genre_name = dirpath_components[-1]
            data["mapping"].append(genre_name)
            print("\nProcessing {}".format(genre_name))

            # alle songs eines genres durchlaufen
            for f in filenames:
                # audiodatei laden
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Audio Features für n_segments berechnen:
                # 1. Zero Crossings
                # 2. Spectral Centroid
                # 3. Spectral Rolloff
                # 4. Chroma Frequencies
                # 5. 13 Mfccs
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s 
                    finish_sample = start_sample + num_samples_per_segment
                    #print(start_sample)
                    #print(finish_sample)

                    # 1. Zero Crossings
                    # - Trackt, wie oft das Signal sein Vorzeichen ändert, also vom negativen ins positive geht
                    # - It usually has higher values for highly percussive sounds like those in metal and rock
                    #zero_crossings = librosa.zero_crossings(x, pad=False)
                    #print(sum(zero_crossings))

                    # 2. Spectral Centroid
                    # - Wo das "Massezentrum" des Sounds vorhanden ist, als gewichteter 
                    #   Durchschnitt der vorhandenden Frequenzen
                    #spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
                    # s_c ist in frames. Diese nun in sekunden umrechnen
                    #frames = range(len(spectral_centroids))
                    #t = librosa.frames_to_time(frames)

                    # 3. Spectral Rolloff
                    #spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
                    #frames = range(len(spectral_rolloff))
                    #t = librosa.frames_to_time(frames)

                    # 5. Chroma Frequencies
                    # - projeziert gesamtes Spektrum auf 12 Halbtöne einer Oktave
                    #hop_length = 512
                    #chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)

                    # 5. Mfcc
                    # Mfcc für ein Segment berechnen. Das Segment wird durch das Intervall [start_sample, finish_sample] gegeben
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length) 

                    # Mfcc Matrix-Transponierte nehmen und diese zur Liste machen. 
                    # Dadurch erhält man (num_samples_per_segment/hop_length) viele Mfcc Vektoren, welche jeweils n_mfcc viele Koeffizienten beinhalten
                    # Bsp: (1292, 13)
                    mfcc = mfcc.T
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

def prepare_cnn_datasets(data_path, test_size):
    """
    Lädt Samples aus .json datei in data_path, splittet diese in
    train, validation und test splits und bereitet die splits
    für das cnn auf, indem die Dimensionen der ndarrays für
    das cnn inputshape angepasst werden.
    """
    # Songdateien laden
    X, Y = load_data(data_path)

    # Train/Test Split erzeugen
    # Die Ergebnisse sind 3D Vekoren -> (num_samples, anzahl mfcc vektoren, n_mfcc koeffizienten)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    # ndarray Dimension für CNN Inputshape anpassen
    # Das Ergebnis sind 4D Vektoren -> (num_samples, anzahl mfcc vektoren, n_mfcc koeffizienten, 1), Bsp: (22, 1292, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_test, Y_train, Y_test


# Main zum Testen der file_processing Funktionen
if __name__ == "__main__":
    
     # Input und Output Ordner
    DATASET_PATH = "../genres_adjusted"
    JSON_PATH = "../data_adjusted_all.json"

    # n_mfcc = 13         -> Anzahl an mfcc koeffizienten
    # n_fft = 2048        -> Breite des Fensters bei der Fourier Transformation
    # hop_length = 512    -> Verschieberate des Fensters
    # num_segments = 10   -> Anzahl Segmente in die ein Song unterteilt wird
    save_mfcc(DATASET_PATH, 
              JSON_PATH, 
              n_mfcc=13,
              n_fft=2048,
              hop_length=512, 
              num_segments=1)

    

    
