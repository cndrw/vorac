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
    count = 0
    
    for file in files:
        mfccs = vru.generate_mfccs_from_textgrid(file, data_dir, phoneme)
        all_mfccs += mfccs
        count += 1
        print(f"File {count}/{total_files} processed for '{phoneme}'")

    return all_mfccs

def train_models(data_dir: Path, phoneme: str) -> dict[str, hmm.GaussianHMM]:
    data = { phoneme: extract_data(data_dir, phoneme) }

    models = {}
    for phoneme, mfccs in data.items():
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(np.vstack(mfccs), lengths=[mfcc.shape[0] for mfcc in mfccs])
        models[phoneme] = model

    return models

training_data = ROOT_DIR / "data" / "198"
models = train_models(training_data, "n")

print("Models trained successfully.")

with open(ROOT_DIR / "models" / "models.pkl", "wb") as f:
    pickle.dump(models, f)
