import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from phoneme import Phoneme

def knn_distance(a, b):
    """Berechnet die euklidische Distanz zwischen zwei Vektoren."""
    sum = 0
    for i in range(len(a)):
        sum += (a[i] - b[i]) ** 2
    return np.sqrt(sum)

def knn_classify(data_set : list, features, k):
    """KNN-Klassifikation basierend auf den gegebenen Features."""
    distances = []
    for i, phoneme in enumerate(data_set):
        distance = knn_distance(phoneme["Amplitude"], features)
        distances.append((distance, f"{i}"))

    # Sortiere nach Distanz und wähle die k nächsten Nachbarn
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    # Zähle die Klassen der Nachbarn
    class_count = {}
    for _, label in neighbors:
        if label in class_count:
            class_count[label] += 1
        else:
            class_count[label] = 1
    
    # Rückgabe der Klasse mit der höchsten Häufigkeit
    return max(class_count.items(), key=lambda x: x[1])[0]


# 1. WAV-Datei laden
samplerate, data = wavfile.read("audios/phoneme/i.wav")

a = Phoneme("a")
i = Phoneme("i")

# 2. Nur ein Kanal verwenden (z. B. linker Kanal bei Stereo)
if len(data.shape) == 2:
    data = data[:, 0]

# 3. FFT anwenden
N = len(data)
T = 1.0 / samplerate  # Abtastintervall
yf = np.fft.fft(data)
xf = np.fft.fftfreq(N, T)

# 4. Nur positive Frequenzen anzeigen
frequenz_mask = (xf >= 0) & (xf <= 2000)
xf_filtered = xf[frequenz_mask]
yf_filtered = np.abs(yf[frequenz_mask])

n = 10
top_indices = np.argsort(yf_filtered)[-n:][::-1]  # sortiert absteigend

yf_filtered = np.sort([yf_filtered[i] for i in top_indices])
xf_filtered = np.sort([xf_filtered[i] for i in top_indices])

res = knn_classify([a.data, i.data], yf_filtered, k=3)
print(res)

# 5. Plotten (optional)
plt.plot(xf_filtered, yf_filtered)
plt.xlabel("Frequenz [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequenzspektrum")
plt.grid()
plt.show()
