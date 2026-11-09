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

class ResearchLabeledDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.samples = []
        self.domain_map = {}
        
        with (self.data_dir / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if row["headcount"] != "unlabeled":
                    layout = row["layout"]
                    if layout not in self.domain_map:
                        self.domain_map[layout] = len(self.domain_map)
                        
                    self.samples.append({
                        "identifier": row["identifier"],
                        "headcount": int(row["headcount"]),
                        "domain_id": self.domain_map[layout]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        csi, ati = load_sample(self.data_dir, sample["identifier"])
        
        start_idx = np.random.randint(0, len(csi) - 600) if len(csi) > 600 else 0
        end_idx = start_idx + 600
        
        frames = csi[start_idx:end_idx].view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
        intervals = ati[start_idx:end_idx].copy()
        intervals[-1] = 0.0
        
        return {
            "frames": torch.from_numpy(frames),
            "intervals": torch.from_numpy(intervals),
            "label": torch.tensor(sample["headcount"], dtype=torch.long),
            "domain": torch.tensor(sample["domain_id"], dtype=torch.long)
        }

class ResearchUnlabeledDataset(Dataset):
    def __init__(self, data_dir: Path, domain_map: dict):
        self.data_dir = data_dir
        self.samples = []
        
        # Trouver l'ID maximum actuel pour continuer l'incrémentation
        current_max_id = max(domain_map.values()) if domain_map else -1
        
        with (self.data_dir / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                layout = row["layout"]
                
                # --- CORRECTION 3 : Attribuer un NOUVEL ID aux domaines inconnus ---
                if layout not in domain_map:
                    current_max_id += 1
                    domain_map[layout] = current_max_id
                
                domain_id = domain_map[layout]
                # -------------------------------------------------------------------
                
                self.samples.append({
                    "identifier": row["identifier"],
                    "domain_id": domain_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        csi, ati = load_sample(self.data_dir, sample["identifier"])
        
        start_idx = np.random.randint(0, max(1, len(csi) - 600))
        end_idx = start_idx + 600
        
        frames_base = csi[start_idx:end_idx].view(np.float32).reshape(600, 1, 3, 1024, 2).copy()
        intervals = ati[start_idx:end_idx].copy()
        intervals[-1] = 0.0
        
        # Weak Augmentation (Identity)[cite: 16]
        frames_w = frames_base.copy()
        
        # Strong Augmentation[cite: 17]
        frames_s = frames_base.copy()
        
        # 1. Random Channel Noise (RCN)
        noise = np.random.normal(0, 0.01, frames_s.shape).astype(np.float32)
        frames_s = frames_s + noise
        
        # 2. Random Subcarrier Masking (RSM) - drop 5%
        num_mask = int(1024 * 0.05)
        mask_indices = np.random.choice(1024, num_mask, replace=False)
        frames_s[:, :, :, mask_indices, :] = 0.0
        
        # 3. Random Gain Scaling (RGS)
        scale = np.random.uniform(0.9, 1.1)
        frames_s = frames_s * scale
        
        return {
            "frames_w": torch.from_numpy(frames_w),
            "frames_s": torch.from_numpy(frames_s),
            "intervals": torch.from_numpy(intervals),
            "domain": torch.tensor(sample["domain_id"], dtype=torch.long)
        }