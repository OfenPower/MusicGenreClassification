import matplotlib.pyplot as plt 
import librosa
import librosa.display
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

def calculate_mel_power_spectrogram(s, sr):
    """Plots the mel power spectrogram."""
    # Mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(s, sr=sr, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S
