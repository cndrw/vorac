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
    model : hmm.GaussianHMM = pickle.load(f)["n"]

mfccs = vru.generate_mfccs_from_textgrid(file="19-198-0027", data_dir=validation_dir, phoneme=u"n")


for mfcc in mfccs:
    log_likelihood = model.score(mfcc, lengths=mfcc.shape[0])
    print(f"Log likelihood for segment: {log_likelihood}")

# 3. modell mit allen segmenten testen
