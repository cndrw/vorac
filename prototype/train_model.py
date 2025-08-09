import matplotlib.pyplot as plt
import numpy as np
import librosa
import pickle
from dataclasses import dataclass
from textgrid import TextGrid
from hmmlearn import hmm
from pathlib import Path

ROOT_DIR = Path(__file__).parent
SAMPLE_RATE = 22050
FFT_SIZE = 512 

@dataclass
class Phoneme:
    start: float
    end: float
    label: str

def read_textgrid(file_path : Path, phoneme : str) -> list[Phoneme]:
    tg = TextGrid()
    tg.read(file_path, encoding='utf-8')

    phonemes = []
    for t in tg.tiers[1]: # read in the phones (1)
        if  t.mark == phoneme:
            phonemes.append(
                Phoneme(start=t.minTime, end=t.maxTime, label=t.mark)
            )
    return phonemes

def extract_audio_segments(file_path: Path, phonemes : list[Phoneme]) -> list[np.ndarray]:
    segments = [] 
    for phoneme in phonemes:
        start, end = phoneme.start, phoneme.end
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, offset=start, duration=end - start)
        segments.append(audio)

    return segments

def convert_to_mfcc(audio_segments: list[np.ndarray]) -> list[np.ndarray]:
    mfccs = []
    for audio in audio_segments:
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, n_mfcc=13)
        mfccs.append(mfcc.T)

    return mfccs

def extract_data(data_dir: Path, phoneme: str) -> list[np.ndarray]:
    all_mfccs = []

    for file in data_dir.iterdir():
        if file.suffix == ".TextGrid":
            phonemes = read_textgrid(file, phoneme)
            audio_segments = extract_audio_segments(file.with_suffix('.flac'), phonemes)
            mfccs = convert_to_mfcc(audio_segments)
            all_mfccs += mfccs

    return all_mfccs

def train_models(data_dir: Path, phoneme: str) -> dict[str, hmm.GaussianHMM]:
    data = { phoneme: extract_data(data_dir, phoneme) }

    models = {}
    for phoneme, mfccs in data.items():
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(np.vstack(mfccs), lengths=[mfcc.shape[0] for mfcc in mfccs])
        models[phoneme] = model

    return models

# 6. Training des Modells mit den MFCC-Features und den Labels

training_data = ROOT_DIR / "data"
models = train_models(training_data, "n")

print("Models trained successfully.")

model_dir = ROOT_DIR / "models"
with open(model_dir / "n_model.pkl", "wb") as f:
    pickle.dump(models['n'], f)
