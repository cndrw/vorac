import pandas as pd

class Phoneme:
    def __init__(self, phoneme: str):
        self.phoneme = phoneme
        self.data = pd.read_csv(f"phoneme/{phoneme}.csv")
