from scipy.io import wavfile
import numpy as np
from pathlib import Path
import csv

global counter, phoneme_amount
counter = 0

def main ():
    global counter, phoneme_amount

    root_dir = Path(__file__).parent
    phoneme_dir = root_dir / "audios" / "phoneme"

    phoneme_sounds = [f for f in phoneme_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
    phoneme_amount = len(phoneme_sounds)

    for phoneme_file in phoneme_sounds:
        convert_wav_to_features(phoneme_file)



def convert_wav_to_features(phoneme_file : Path): 
    global counter, phoneme_amount
    samplerate, data = wavfile.read(phoneme_file)

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

    # save as csv
    filename = phoneme_file.stem  # Use the stem of the Path object
    with open(f"phoneme/{filename}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frequency", "Amplitude"])
        for freq, amp in zip(xf_filtered, yf_filtered):
            writer.writerow([freq, amp])

    counter += 1
    print(f"Feature extraction completed ({counter}/{phoneme_amount}). Results saved to {filename}.csv")


if __name__ == "__main__":
    main()
