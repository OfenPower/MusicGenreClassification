import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn

print("--------------------------------------------------------------")

# Allg. Infos
# - The sample rate is the number of samples of audio carried per second, measured in Hz or kHz.

# Audio Datei einlesen
audio_path = '../genres/blues/blues.00001.wav'
x , sr = librosa.load(audio_path)               # type(x)=numpy.ndarray, type(sr) = int


# 1. Zero Crossings
# - Trackt, wie oft das Signal sein Vorzeichen ändert, also vom negativen ins positive geht
#zero_crossings = librosa.zero_crossings(x, pad=False)
#print(sum(zero_crossings))

# Air = 758292
# ScreamAimFire = 819414
# ToZanarkand = 235520
# LittleBadGirl = 869708


# 2. Spectral Centroid
# - Wo das "Massezentrum" des Sounds vorhanden ist, als gewichteter 
#   Durchschnitt der vorhandenden Frequenzen

#spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0] 
#print(spectral_centroids.shape)
#print(spectral_centroids)

# s_c ist in frames. Diese nun in sekunden umrechnen
#frames = range(len(spectral_centroids))
#t = librosa.frames_to_time(frames)

# Normalisierung für Visualisierung
#def normalize(x, axis=0):
#    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Spectral Centroids entlang der Waveform plotten
#librosa.display.waveplot(x, sr=sr, alpha=0.4)
#plt.plot(t, normalize(spectral_centroids), color="r")
#plt.show()


# 3. Spectral Rolloff
#

#spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
#frames = range(len(spectral_rolloff))
#t = librosa.frames_to_time(frames)

#librosa.display.waveplot(x, sr=sr, alpha=0.4)
#plt.plot(t, normalize(spectral_rolloff), color='r')
#plt.show()

# 4. MFCC

#mfccs = librosa.feature.mfcc(x, sr=sr)
#mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
#plt.figure(figsize=(14,4))
#librosa.display.specshow(mfccs, sr=sr, x_axis="time")
#plt.show()



# 5. Chroma Frequencies
# - projeziert gesamtes Spektrum auf 12 Halbtöne einer Oktave

hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
print(chromagram.shape)
plt.figure(figsize=(15,5))
librosa.display.specshow(chromagram, 
                        x_axis='time', 
                        y_axis='chroma', 
                        hop_length=hop_length, 
                        cmap='coolwarm')
plt.show()
