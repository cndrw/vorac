import numpy as np
import librosa
from textgrid import TextGrid
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent
SAMPLE_RATE = 22050

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
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
        mfccs.append(mfcc)
    return mfccs


def train_model(data_dir: Path, phoneme: str) -> None:
    i = 0
    for file in data_dir.iterdir():
        if file.suffix == ".TextGrid" and i < 1:
            i += 1
            phonemes = read_textgrid(file, phoneme)
            audio_segments = extract_audio_segments(file.with_suffix('.flac'), phonemes)
            mfccs = convert_to_mfcc(audio_segments)


# 5. MFCC values labeln mit dem phonem
# 6. Training des Modells mit den MFCC-Features und den Labels

training_data = ROOT_DIR / "data"
train_model(training_data, "n")
