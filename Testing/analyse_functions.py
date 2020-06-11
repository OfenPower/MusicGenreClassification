import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy
import sklearn
import numpy as np

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


def plot_wave(y, sr):
    """Plots the waveform"""
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.show() 

def plot_spectrogram(y, sr):
    """Plots the spectogramm"""
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(np.abs(X))
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')        # y_axis='hz'
    plt.colorbar()
    plt.show()

# Funktion zum Plotten des Betragsspektrums der Fourier-Transformation
def plot_fft(audio, sampling_rate):
    """Plots the existing frequencies"""
    n = len(audio)
    T = 1 / sampling_rate       # Periodendauer
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0 * T), n//2)
    
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))  # Nur die erste Hälfte an Werten betrachten 
    plt.grid()
    plt.xlabel("Frequency --->")
    plt.ylabel("Magnitude")
    return plt.show()


def plot_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    plt.figure(figsize=(14,4))
    librosa.display.specshow(mfccs, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar()
    plt.show()

def calc_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(x, sr=sr)
    return mfccs

# -- Audio Datei einlesen -- #
#audio_path = './AudioFiles/AveMaria.wav'
audio_path = "C:/Users/Ofen/Documents/Hochschule Master/1. Semester/MachineLearning/AudioData/genres/genres/reggae/reggae.00018.wav"
#audio_path = "C:/Users/Ofen/Documents/Hochschule Master/1. Semester/MachineLearning/AudioData/genres/genres/pop/pop.00000.wav"
x , sr = librosa.load(audio_path)

#plot_wave(x, sr)
#plot_fft(x, sr)
#plot_spectrogram(x, sr)
#plot_mel_power_spectrogram(x, sr)
#plot_mfcc(x, sr)





def plot_MFCC_coeffs(y, sr):
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128) # shape = (128, ???)
	log_S = librosa.power_to_db(S, ref=np.max)

	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13) # shape = (13, ???)

	delta2_mfcc = librosa.feature.delta(mfcc, order=2) # shape = (13, ???)

	plt.figure(figsize=(12, 4))
	librosa.display.specshow(delta2_mfcc)
	plt.ylabel('MFCC coeffs')
	plt.xlabel('Time')
	plt.title('MFCC')
	plt.colorbar()
	plt.tight_layout()
	plt.show()
	return

#plot_MFCC_coeffs(x, sr)