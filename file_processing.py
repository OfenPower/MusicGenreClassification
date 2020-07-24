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
        "spectralcentroid": [],
        "spectralrolloff": [],
        "chromafrequencies": [],
        "mfccs": [],
        "labels": []
    }

    # Samples per Segment für einen Track mittels num_segments berechnen
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    
    # mittels der hop_length die anzahl an ergebnis mfcc-vektoren pro samples in einem segment berechnen
    expected_num_feature_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

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
                # 1. Spectral Centroid
                # 2. Spectral Rolloff
                # 3. Chroma Frequencies
                # 4. 13 Mfccs

                # Features eines Songs für jedes Songsegment berechnen
                for s in range(num_segments):
                    # flags um festzustellen, ob Berechnung der Features erfolgreich war
                    spectral_centroid_success = False
                    spectral_rolloff_success = False
                    chroma_frequencies_success = False
                    mfccs_success = False
                        
                    # Ein Segment wird durch das Intervall [start_sample, finish_sample] gegeben
                    # Wenn n_segments=1, dann ist das Intervall [0, SAMPLES_PER_TRACK], welches den ganzen 30s Song enthält
                    start_sample = num_samples_per_segment * s 
                    finish_sample = start_sample + num_samples_per_segment
                    #print(start_sample)
                    #print(finish_sample)

                    # 1. Spectral Centroid
                    # - Wo das "Massezentrum" des Sounds vorhanden ist, als gewichteter 
                    #   Durchschnitt der vorhandenden Frequenzen
                    # Bso: "So spectral centroid for blues song will lie somewhere 
                    #       near the middle of its spectrum while that for a metal 
                    #       song would be towards its end"
                    spectral_centroids = librosa.feature.spectral_centroid(signal[start_sample:finish_sample], 
                                                                            sr=sr,
                                                                            n_fft=n_fft,
                                                                            hop_length=hop_length)[0] # -> [0] wird benötigt, weil das Ergebnis (1, t) ist 
                                                                                                      #    und wir an t interessiert sind
                    # spectral_centroids ist in frames. Diese nun in Sekunden umrechnen
                    frames = range(len(spectral_centroids))
                    s_c = librosa.frames_to_time(frames)
                    print(s_c.shape)
                    # Spectral Centroid Liste als Feature aufnehmen
                    if len(s_c) == expected_num_feature_vectors_per_segment:
                        spectral_centroid_success = True
                        s_c_list = s_c.tolist()
                        data["spectralcentroid"].append(s_c_list)
                    
                    
                    

                    # 2. Spectral Rolloff
                    # "It represents the frequency below which a specified percentage of the total spectral energy, here 85%, lies"
                    spectral_rolloff = librosa.feature.spectral_rolloff(signal[start_sample:finish_sample], 
                                                                        sr=sr,
                                                                        n_fft=n_fft,
                                                                        hop_length=hop_length)[0] # -> [0] wird benötigt, weil das Ergebnis (1, t) ist 
                                                                                                      #    und wir an t interessiert sind
                    # spectral_rolloff ist in frames. Diese nun in Sekunden umrechnen
                    frames = range(len(spectral_rolloff))
                    s_r = librosa.frames_to_time(frames)
                    print(s_r.shape)
        	        # spectral_rolloff als Liste anhängen
                    if len(s_r) == expected_num_feature_vectors_per_segment:
                        spectral_rolloff_success = True
                        s_r_list = s_r.tolist()
                        data["spectralrolloff"].append(s_r_list)
                    

                    # 3. Chroma Frequencies
                    # - projeziert gesamtes Spektrum auf 12 Halbtöne einer Oktave
                    # - Bsp-Ergebnisshape: (12, 1293)
                    chromagram = librosa.feature.chroma_stft(signal[start_sample:finish_sample], 
                                                                sr=sr,
                                                                n_fft=n_fft,
                                                                hop_length=hop_length)
                    # Chroma Frequencies Transpose als Liste anhängen
                    chromagram = chromagram.T
                    print(chromagram.shape)
                    if len(chromagram) == expected_num_feature_vectors_per_segment:
                        chroma_frequencies_success = True
                        chromagram_list = chromagram.tolist()
                        data["chromafrequencies"].append(chromagram_list)       

                    # 4. Mfcc
                    # 13 Mfcc berechnen
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_mfcc=n_mfcc, 
                                                    n_fft=n_fft, 
                                                    hop_length=hop_length) 
                    # Mfcc Matrix-Transponierte nehmen und diese zur Liste machen. 
                    # Dadurch erhält man (num_samples_per_segment/hop_length) viele Mfcc Vektoren, welche jeweils n_mfcc viele Koeffizienten beinhalten
                    # Bsp: (1292, 13)
                    mfcc = mfcc.T
                    print(mfcc.shape)
                    if len(mfcc) == expected_num_feature_vectors_per_segment:
                        mfccs_success = True
                        data["mfccs"].append(mfcc.tolist())
                    
                    # label anhängen, falls alle Features erfolgreich berechnet wurden
                    if spectral_centroid_success and spectral_rolloff_success and chroma_frequencies_success and mfccs_success:
                        # Label anhängen
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))

    # json Datei anlegen
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Funktion zum Laden einer .json Datei des Datasets
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # listen aus json in ndarrays umwandeln
    spectralcentroid = np.array(data["spectralcentroid"])
    spectralrolloff = np.array(data["spectralrolloff"])
    chromafrequencies = np.array(data["chromafrequencies"])
    mfccs = np.array(data["mfccs"])
    labels = np.array(data["labels"])

    # alle features in ein 3D Inputarray einbetten, das wie folgt aussieht:
    # (x, y, z) mit
    # x = Anzahl Datensätze
    # y = (num_samples_per_segment / hop_length) - viele Datenabschnitte z.B. 1292 
    # z = Anzahl aller Features pro Datenabschnitt, also 13Mfccs + 12chroma + 1s_c + 1s_r + 1zero_crossing = 28 features
    # Bsp: (22, 1292, 28)
    # Damit die Einbettung funktioniert, müssen alle Feature-Arrays in 3d gewandelt werden, damit alle konkateniert werden können

    #print(zerocrossings.shape)
    #print(spectralcentroid.shape)
    #print(spectralrolloff.shape)
    #print(chromafrequencies.shape)
    #print(mfccs.shape)

    # spectralcentroid und spectralrolloff um 3. Dimension erweitern und konkatenieren
    #print("--- concat s_c, s_r ---")
    spectralcentroid3d = spectralcentroid[..., np.newaxis]     # s_c array um 3. Dimension erweitern
    spectralrolloff3d = spectralrolloff[..., np.newaxis]     # s_r array um 3. Dimension erweitern
    rows = spectralcentroid.shape[0]
    cols = spectralcentroid.shape[1]
    #print("rows: {}, cols: {}".format(rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            spectralcentroid3d[i, j, 0] = spectralcentroid[i, j]    # -> s_c Werte in 3. Dimension schreiben
            spectralrolloff3d[i, j, 0] = spectralrolloff[i, j]     # -> s_r Werte in 3. Dimension schreiben
    # s_c und s_r in dritter Dimension konkatenieren
    inputs1 = np.concatenate((spectralcentroid3d, spectralrolloff3d), axis=2)

    # Nun werden chroma und mfcc konkateniert. Da diese schon in 3d sind, geht das ohne 3D Vorverarbeitung
    #print("--- concat Chroma and Mfcc ---")
    chrom_mfcc = np.concatenate((chromafrequencies, mfccs), axis=2)

    # chrom_mfcc an inputs2 konkatenieren. Das Ergebnis hat 28 Features in der 3. Dimension
    #print("--- last concat ---")
    inputs2 = np.concatenate((inputs1, chrom_mfcc), axis=2)

    # DEBUG Prints zur manuellen Überprüfung der Werte mit der dazugehörigen .json Datei 
    print(inputs2.shape)
    #print(inputs3[0, 0, 0]) # -> ZeroCrossing Wert des ersten Samples
    #print(inputs3[0, 0, 1]) # -> s_c Wert des ersten Samples
    #print(inputs3[0, 0, 2]) # -> s_r Wert des ersten Samples
    #print(inputs3[0, 0, 3]) # -> Erster chroma Wert des ersten Samples
    #print(inputs3[0, 0, 15]) # -> Erster Mfcc Wert des ersten Samples

    print("Songs successfully loaded!")

    return inputs2, labels

# Funktion zum Laden einer .json Datei des Datasets, die für den Einsatz in einem CNN aufbereitet wird
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
    JSON_PATH = "../data_adjusted_all_n10.json"
    #JSON_PATH = "../data_adjusted_n1.json"

    # n_mfcc = 13         -> Anzahl an mfcc koeffizienten
    # n_fft = 2048        -> Breite des Fensters bei der Fourier Transformation (STFT)
    # hop_length = 512    -> Verschieberate des Fensters
    # num_segments = 10   -> Anzahl Segmente in die ein Song unterteilt wird
    save_mfcc(DATASET_PATH,  
              JSON_PATH, 
              n_mfcc=13,
              n_fft=2048,
              hop_length=512, 
              num_segments=10)

    print("Done")

    #inputs, labels = load_data(JSON_PATH)
    #X1, Y1, X2, Y2 = prepare_cnn_datasets(JSON_PATH, 0.25)

    #print("SUCCESS!!!")
    #print(X1.shape)
    #print(Y1.shape)

    

    
