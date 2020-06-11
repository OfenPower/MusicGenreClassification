import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import numpy as np

def plot_mfcc(y, sr, mfcc_amount=20, lift=0):
    mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=mfcc_amount, lifter=lift)
    plt.figure(figsize=(14,4))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar()
    plt.show()

def calc_mfcc(y, sr, mfcc_amount=20, lift=0):
    mfccs = librosa.feature.mfcc(x, sr=sr, n_mfcc=mfcc_amount, lifter=lift)
    return mfccs

def plot_mel_power_spectrogram(s, sr):
    """Plots the mel power spectrogram."""

    # Mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(s, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

    # Plot params
    plt.colorbar(format='%+02.0f dB')
    plt.title('mel power spectrogram')
    plt.tight_layout()
    plt.show() 


# .) Was ist sample_rate?
# sampling_rate = 16k says that this audio was recorded(sampled) 
# with a sampling frequency of 16k. In other words, while recording this 
# file we were capturing 16000 amplitudes every second

# .) Mfcc shape = (n_mfcc, t). Was ist t?
# ------------------------------------
# By Default, the Mel-scaled power spectrogram window and hop length are the following:
#   n_fft=2048
#   hop_length=512 (number of samples between successive frames)
# So assuming you used the default sample rate (sr=22050), the output of your mfcc function makes sense:
#   output length = (seconds) * (sample rate) / (hop_length) 
# ANDERS FORMULIERT:
# ------------------------------------
# By printing the shape of mfccs you get how many mfccs are calculated on how many frames. 
# The first value represents the number of mfccs calculated and another value represents a number of frames available

# .) Merke: Lifter = 2*MFCC Count für gutes Ergebnis!



if __name__ == "__main__"

# -- Audio Datei einlesen -- 
#audio_path = './AudioFiles/AveMaria.wav'
#audio_path = "C:/Users/Ofen/Documents/Hochschule Master/1. Semester/MachineLearning/AudioData/genres/genres/reggae/reggae.00018.wav"
#audio_path = "C:/Users/Ofen/Documents/Hochschule Master/1. Semester/MachineLearning/AudioData/genres/genres/pop/pop.00000.wav"
    audio_path = "C:/Users/Ofen/Documents/Hochschule Master/1. Semester/MachineLearning/AudioData/genres/genres/speech/speech01.wav"
    x , sr = librosa.load(audio_path)


    plot_mfcc(x, sr)
#plot_mfcc(x, sr, 20, 40)
    plot_mel_power_spectrogram(x, sr)
#y2 = calc_mfcc(x, sr, 20, 40)
#print(y1.shape)
#print(y2.shape)
