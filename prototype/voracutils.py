import numpy as np
import librosa
from dataclasses import dataclass
from textgrid import TextGrid
from pathlib import Path

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

def generate_mfccs_from_textgrid(file: str, data_dir: Path, phoneme: str) -> list[np.ndarray]:
    phonemes = read_textgrid(data_dir / "output" / f"{file}.TextGrid", phoneme)
    audio_segments = extract_audio_segments(data_dir / f"{file}.flac", phonemes)
    mfccs = convert_to_mfcc(audio_segments)
    return mfccs