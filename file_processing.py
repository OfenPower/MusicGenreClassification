import os
import numpy as np
import librosa
import math
import json

# Input und Output Ordner
DATASET_PATH = "../genres"
JSON_PATH = "../data_all_10.json"

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
              num_segments=1):  # songs in segmente aufteilen. Für jedes Segment werden n_mffcs berechnet. Dadurch wird die TrainingData um num_segments erweitert
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

# Main
if __name__ == "__main__":
    """ save_mfcc(DATASET_PATH, 
              JSON_PATH, 
              n_mfcc=13,  
              n_fft=2048, 
              hop_length=512, 
              num_segments=1) """

    inputs, targets = load_data(JSON_PATH)

    print(len(inputs))