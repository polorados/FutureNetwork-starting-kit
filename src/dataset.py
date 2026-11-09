# This file is responsible for loading the dataset and providing it to the model during training and validation.
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
            
            # Extract a 600-frame window
            if self.is_training and len(csi) > 600:
                start_idx = np.random.randint(0, len(csi) - 600)
            else:
                start_idx = 0
                
            end_idx = start_idx + 600
            
            csi_window = csi[start_idx:end_idx]
            ati_window = ati[start_idx:end_idx].copy()
            ati_window[-1] = 0.0
            
            # Convert complex64 to two float32 values (real and imaginary)
            frames = csi_window.view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
            
            # ==========================================
            # 🚀 DATA AUGMENTATION (Training Only)
            # ==========================================
            if self.is_training:
                # 1. Gaussian Noise (Applies 50% of the time)
                # Prevents memorizing exact signal magnitudes by adding slight static
                if np.random.rand() < 0.5:
                    # std=0.01 is a starting point; adjust if the noise is too strong
                    noise = np.random.normal(0, 0.01, frames.shape).astype(np.float32)
                    frames = frames + noise

                # 2. Subcarrier Masking / Frequency Dropout (Applies 50% of the time)
                # Forces the model to look at the whole spectrum, not just specific subcarriers 
                # that might be uniquely distorted by the training room's furniture
                if np.random.rand() < 0.5:
                    # Mask out 5% of the 1024 subcarriers
                    num_mask = int(1024 * 0.05)
                    mask_indices = np.random.choice(1024, num_mask, replace=False)
                    # Set those specific subcarriers to 0 across all time, antennas, and components
                    frames[:, :, :, mask_indices, :] = 0.0

                # 3. Frame Dropping / Time Masking (Applies 50% of the time)
                # Simulates network packet loss and prevents the LSTM from memorizing exact walking speeds
                if np.random.rand() < 0.5:
                    # Drop 5% of the 600 frames
                    num_drop = int(600 * 0.05)
                    drop_indices = np.random.choice(600, num_drop, replace=False)
                    frames[drop_indices, :, :, :, :] = 0.0
            # ==========================================

            return {
                "frames": torch.from_numpy(frames),
                "intervals": torch.from_numpy(ati_window),
                "label": torch.tensor(sample_info["headcount"], dtype=torch.long)
            }
