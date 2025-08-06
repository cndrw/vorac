import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import librosa
from collections import deque
import time

# TODO:
#    - convert regular spectrogram to mel spectrogram 

RATE = 22050
CHUNK = 1024
ROLLING_SECONDS = 2
N_FFT = 512
HOP_LENGTH = 256

audio = pyaudio.PyAudio()

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

audio_buffer = np.zeros(RATE * ROLLING_SECONDS, dtype=np.float32)

# Matplotlib vorbereiten
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
img = None

print("Live-STFT startet... Beende mit STRG+C")

chunk_counter = 0
spec_buffer = deque(maxlen=5)
vmin = 0  
vmax = 150

try:
    while True:
        data = stream.read(1024)

        audio_buffer = np.roll(audio_buffer, -CHUNK)
        audio_buffer[-CHUNK:] = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        chunk_counter += 1 

        if chunk_counter < RATE / CHUNK: 
            continue
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio_buffer[len(audio_buffer) // 2:],
            sr=RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=40)

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        spec_buffer.append(mel_spec_db)

        ax.clear()
        img = librosa.display.specshow(
            np.concatenate(spec_buffer, axis=1),
            sr=RATE,
            # hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            ax=ax,
            )
        ax.set_title('Live Spectrogram')

        plt.pause(0.01)
        chunk_counter = 0

except KeyboardInterrupt:
    pass


stream.stop_stream()
stream.close()
audio.terminate()
plt.ioff()