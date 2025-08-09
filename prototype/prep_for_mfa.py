from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--directory", "-d", type=Path, default="Error", help="Path to the .trans.txt file.")
args = parser.parse_args()

file_path = Path(args.directory)
dir_path = file_path.parent

if str(file_path) == "Error":
    print("Please provide a directory path using --directory or -d argument.")
    exit(1)

with open(file_path, "r") as f:
    lines = f.readlines()

for line in lines:
    name, text = line.split(" ", 1)
    with open(dir_path / f"{name}.txt", "w") as f:
        f.write(text.strip())

print(f"All files created successfully in {dir_path}.")
