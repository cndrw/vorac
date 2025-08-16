from argparse import ArgumentParser
from hmmlearn import hmm
from pathlib import Path
import voracutils as vru
import pickle

ROOT = Path(__file__).parent

parser = ArgumentParser()
parser.add_argument("--directory", "-d", type=Path, default="Error")
args = parser.parse_args()

if str(args.directory) == "Error":
    print("Please provide a directory with the --directory or -d argument.")
    exit(1)

validation_dir = Path(args.directory)

with open(ROOT / "models" / "models.pkl", "rb") as f:
    models : hmm.GaussianHMM = pickle.load(f)

files = sorted(set([file.stem for file in validation_dir.iterdir()]))[:-2] # exlude output dir and .trans file

phonemes = list(models.keys())
tests = { p: [] for p in phonemes }
for phoneme in phonemes:
    for file in files:
        features = vru.generate_features_from_textgrid(file=file, data_dir=validation_dir, phoneme=phoneme)
        tests[phoneme] += features

print(f"Testing with {len(files)} files")

total_passed = 0
for phoneme, features in tests.items():
    passed_counter = 0
    for feature in features:
        scores = { phoneme: model.score(feature) for phoneme, model in models.items() }
        best_phoneme = max(scores, key=scores.get)

        passed_counter += int(best_phoneme == phoneme)
    total_passed += passed_counter

    print(f"Testing '{phoneme}': Passed {passed_counter}/{len(features)} {'️️✔️' if passed_counter > (len(features) // 2) else '❌'}")

passed_percentage = total_passed / sum(len(features) for features in tests.values()) * 100
print(f"Result: {passed_percentage:.2f}% of tests passed")
