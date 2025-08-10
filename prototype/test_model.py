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

all_tests = []
for file in files:
    mfccs = vru.generate_mfccs_from_textgrid(file=file, data_dir=validation_dir, phoneme=u"p")
    all_tests.append(mfccs)

total_test = sum(len(mfcc) for mfcc in all_tests)
print(f"Testing on {len(all_tests)} files with {total_test} occurences...")

count = 0
for i, mfcc in enumerate(all_tests):
    for j, segment in enumerate(mfcc):
        scores = { phoneme: model.score(segment) for phoneme, model in models.items() }
        best_phoneme = max(scores, key=scores.get)
        count += 1
        print(f"Best fit [{count:03}/{total_test}]: {best_phoneme}")
