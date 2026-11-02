import csv
import torch
import blosc2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

def load_sample(folder, identifier):
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati

class RoomSenseDataset(Dataset):
    def __init__(self, data_dir: Path, is_training=True):
        self.data_dir = data_dir
        self.is_training = is_training
        self.samples = []

        # Lire le samples.csv pour récupérer les identifiers le label (headcount) associé
        with (self.data_dir / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                # si les données ne sont pas labelisées on met le headcount à -1
                headcount = -1 if row["headcount"] == "unlabeled" else int(row["headcount"])
                self.samples.append({
                    "identifier":row["identifier"],
                    "headcount": headcount
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        csi, ati = load_sample(self.data_dir, sample_info["identifier"])

        # On extrait une fenetre de 600 frames
        if self.is_training and len(csi) > 600:
            # On recupère 600 frames aléatoires pour l'entrainement
            start_idx = np.random.randint(0, len(csi) - 600)
        else: 
            # On prend les premiers 600 frames pour de la validation
            start_idx = 0

        end_idx = start_idx + 600

        csi_window = csi[start_idx:end_idx]
        ati_window = ati[start_idx:end_idx].copy()
        ati_window[-1] = 0.0

        # On converti les données de type complex64 en données de type float32 (real and imaginary)
        frames = csi_window.view(np.float32).reshape(600, 1, 3, 1024, 2)

