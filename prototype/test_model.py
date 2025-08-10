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

PHONEMES = ['n', 's', 'p']

tests = { p: [] for p in PHONEMES }
for phoneme in PHONEMES:
    for file in files:
        mfccs = vru.generate_mfccs_from_textgrid(file=file, data_dir=validation_dir, phoneme=phoneme)
        tests[phoneme].append(mfccs)

# tests[phoneme][file][phoneme occurrence] = mfcc

total_test = sum(len(mfcc) for mfccs in tests.values() for mfcc in mfccs)
print(f"Testing with {len(files)} files")

fails = []
# Testing '$phoneme' [fileIdx/files][currTestIdx/allPhonemesOccurences] : $checkmark
for phoneme, all_mfccs in tests.items():
    res = []
    for i, mfccs in enumerate(all_mfccs):
        for j, segment in enumerate(mfccs):
            scores = { phoneme: model.score(segment) for phoneme, model in models.items() }
            best_phoneme = max(scores, key=scores.get)

            checkmark = '✔️' if best_phoneme == phoneme else '❌'
            res.append(checkmark)
            if checkmark == '❌':
                fails.append((phoneme, i, j, best_phoneme))
    print(f"Testing '{phoneme}' : {'️️✔️' if all(checkmark == '✔️' for checkmark in res) else '❌'}")


print("-" * 30)
for phoneme, i, j, best_phoneme in fails:
    print(f"Failed: Testing '{phoneme}' [{(i+1):02}/{len(tests[phoneme]):02}][{j+1:02}/{len(tests[phoneme][i]):02}] : ❌")
