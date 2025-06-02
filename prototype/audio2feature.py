from scipy.io import wavfile
from argparse import ArgumentParser
import numpy as np
import csv

# Argumente parsen
parser = ArgumentParser(description="Audio to Feature Extraction")
parser.add_argument("--file", "-f", type=str, default="ERROR", help="Path to the audio file")
args = parser.parse_args()

if args.file == "ERROR":
    print("Error: Please provide a valid audio file path using --file or -f argument.")
    exit(1)

samplerate, data = wavfile.read(args.file)

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

# Indizes der n größten Werte
n = 10
top_indices = np.argsort(yf_filtered)[-n:][::-1]  # sortiert absteigend

yf_filtered = np.sort([yf_filtered[i] for i in top_indices])
xf_filtered = np.sort([xf_filtered[i] for i in top_indices])

# speichern in csv datei
filename = args.file.split("/")[-1].split(".")[0]  # Extract filename without extension
with open(f"phoneme/{filename}.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frequency", "Amplitude"])
    for freq, amp in zip(xf_filtered, yf_filtered):
        print(f"Frequency: {freq:.2f}, Amplitude: {amp:.2f}")
        writer.writerow([freq, amp])

print(f"Feature extraction completed. Results saved to {filename}.csv")