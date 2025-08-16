import voracutils as vru
import numpy as np
import pickle
from hmmlearn import hmm
from pathlib import Path

ROOT_DIR = Path(__file__).parent

def extract_data(data_dir: Path, phoneme: str) -> list[np.ndarray]:
    feature_collection = []
    files = sorted(set([file.stem for file in data_dir.iterdir()]))[:-2] # exlude output dir and .trans file

    total_files = len(files)
    ending = '\r'

    for i, file in enumerate(files):
        features = vru.generate_features_from_textgrid(file, data_dir, phoneme)
        feature_collection += features

        if i == total_files - 1:
            ending = '\n'
        print(f"File {i + 1}/{total_files} processed for '{phoneme}' ({len(feature_collection)} samples)", end=ending)

    return feature_collection

def train_models(data_dir: Path, phoneme: list[str]) -> dict[str, hmm.GaussianHMM]:
    data = { p: extract_data(data_dir, p) for p in phoneme }

    min_samples = 20
    for key in list(data.keys()):
        if len(data[key]) < 20:
            del data[key]
            print(f"Removed phoneme '{key}' due to less than {min_samples} data samples.")

    models = {}
    for phoneme, features in data.items():
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(np.vstack(features), lengths=[feature.shape[0] for feature in features])
        models[phoneme] = model

    return models

training_data = ROOT_DIR / "data" / "198"

PHONEMES = [
    u"", u"aj", u"aw", u"b", u"bʲ", u"c", u"cʰ", u"cʷ", u"d", u"dʒ", u"dʲ",
    u"d̪", u"ej", u"f", u"fʲ", u"h", u"i", u"iː", u"j", u"k", u"kʰ",
    u"kʷ", u"l", u"m", u"mʲ", u"m̩", u"n", u"n̩", u"ow", u"p", u"pʰ",
    u"pʲ", u"pʷ", u"s", u"t", u"tʃ", u"tʰ", u"tʲ", u"tʷ", u"t̪", u"v",
    u"vʲ", u"w", u"z", u"æ", u"ç", u"ð", u"ŋ", u"ɐ", u"ɑ", u"ɑː",
    u"ɒ", u"ɒː", u"ɔj", u"ə", u"ɚ", u"ɛ", u"ɝ", u"ɟ", u"ɟʷ", u"ɡ",
    u"ɡʷ", u"ɪ", u"ɫ", u"ɫ̩", u"ɱ", u"ɲ", u"ɹ", u"ɾ", u"ɾʲ", u"ɾ̃",
    u"ʃ", u"ʉ", u"ʉː", u"ʊ", u"ʎ", u"ʒ", u"ʔ", u"θ"
]



models = train_models(training_data, PHONEMES)

print("Models trained successfully.")

with open(ROOT_DIR / "models" / "models.pkl", "wb") as f:
    pickle.dump(models, f)
