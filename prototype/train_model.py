import matplotlib.pyplot as plt
import numpy as np
import voracutils as vru
import pickle
from dataclasses import dataclass
from textgrid import TextGrid
from hmmlearn import hmm
from pathlib import Path

ROOT_DIR = Path(__file__).parent

def extract_data(data_dir: Path, phoneme: str) -> list[np.ndarray]:
    all_mfccs = []
    files = sorted(set([file.stem for file in data_dir.iterdir()]))[:-2] # exlude output dir and .trans file

    total_files = len(files)
    ending = '\r'

    for i, file in enumerate(files):
        mfccs = vru.generate_mfccs_from_textgrid(file, data_dir, phoneme)
        all_mfccs += mfccs

        if i == total_files - 1:
            ending = '\n'
        print(f"File {i + 1}/{total_files} processed for '{phoneme}'", end=ending)

    return all_mfccs

def train_models(data_dir: Path, phoneme: list[str]) -> dict[str, hmm.GaussianHMM]:
    data = { p: extract_data(data_dir, p) for p in phoneme }

    models = {}
    for phoneme, mfccs in data.items():
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(np.vstack(mfccs), lengths=[mfcc.shape[0] for mfcc in mfccs])
        models[phoneme] = model

    return models

training_data = ROOT_DIR / "data" / "198"

# list with all phonemes
# PHONEMES = ['n', 's', 'p']

# PHONEMES = [
#     # Consonants
#     u"p", u"b", u"t", u"d", u"k", u"g",
#     u"tʃ", u"dʒ", u"f", u"v", u"θ", u"ð",
#     u"s", u"z", u"ʃ", u"ʒ", u"h", u"m",
#     u"n", u"ŋ", u"l", u"r", u"j", u"w",

#     # Vowels (monophthongs)
#     u"i", u"ɪ", u"e", u"ɛ", u"æ",
#     u"ɑ", u"ɒ", u"ɔ", u"ʌ", u"ʊ", u"u", u"ɝ", u"ɚ", u"ə",

#     # Diphthongs
#     u"aɪ", u"aʊ", u"ɔɪ", u"eɪ", u"oʊ"
# ]

models = train_models(training_data, PHONEMES)

print("Models trained successfully.")

with open(ROOT_DIR / "models" / "models.pkl", "wb") as f:
    pickle.dump(models, f)
