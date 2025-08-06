from textgrid import TextGrid
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Phoneme:
    start: float
    end: float
    label: str


def read_textgrid(file_path : Path, phoneme : str):
    tg = TextGrid()
    tg.read(file_path, encoding='utf-8')

    for t in tg.tiers[1]: # read in the phones (1)
        if  t.mark == phoneme:
            feat = Phoneme(
                start=t.minTime,
                end=t.maxTime,
                label=t.mark
            )

    return feat

# 1. auslesen der TextGrid-Datei
# 2. extrahieren aller Intervalle für das vorgegebene phonem
# 3. extrahieren der zugehörigen Audiodaten
# 4. MFCC der audiosegmente berechnen
# 5. MFCC values labeln mit dem phonem
# 6. Training des Modells mit den MFCC-Features und den Labels

ROOT_DIR = Path(__file__).parent

read_textgrid(ROOT_DIR / "data" / "19-198-0001.TextGrid", u"n")
