# This file is responsible for loading the dataset and providing it to the model during training and validation.
import csv
import torch
import blosc2
import numpy as np
from functools import lru_cache
from pathlib import Path
from torch.utils.data import Dataset

WINDOW = 600  # frames per validation/test sample


def load_sample(folder, identifier):
    csi = blosc2.load_array(str(folder / f"CSI_{identifier}.b2"))
    ati = blosc2.load_array(str(folder / f"ATI_{identifier}.b2"))
    return csi, ati


@lru_cache(maxsize=2)
def _load_sample_cached(folder_str, identifier):
    # Used in evaluation mode only: several windows are taken from the same 12,000-frame
    # trace, so we avoid decompressing it again for every window. Never modify these arrays.
    return load_sample(Path(folder_str), identifier)


class RoomSenseDataset(Dataset):
    def __init__(self, data_dir: Path, is_training=True, windows_per_trace=4):
        """
        is_training=True : one random 600-frame window per trace, with augmentation.
        is_training=False: `windows_per_trace` fixed, evenly spaced windows per trace,
                           so len(dataset) = n_traces * windows_per_trace and
                           index i -> (trace i // K, window i % K).
        """
        self.data_dir = Path(data_dir)
        self.is_training = is_training
        self.windows_per_trace = 1 if is_training else windows_per_trace
        self.samples = []

        with (self.data_dir / "samples.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                headcount = -1 if row["headcount"] == "unlabeled" else int(row["headcount"])
                self.samples.append({
                    "identifier": row["identifier"],
                    "layout": row.get("layout", "unknown"),
                    "room": row.get("room", "unknown"),
                    "headcount": headcount,
                })

    @property
    def n_traces(self):
        return len(self.samples)

    def __len__(self):
        return len(self.samples) * self.windows_per_trace

    def __getitem__(self, idx):
        trace_idx, k = divmod(idx, self.windows_per_trace)
        sample_info = self.samples[trace_idx]

        if self.is_training:
            csi, ati = load_sample(self.data_dir, sample_info["identifier"])
        else:
            csi, ati = _load_sample_cached(str(self.data_dir), sample_info["identifier"])

        n = len(csi)
        assert n >= WINDOW, f"Trace {sample_info['identifier']} has only {n} frames"
        max_start = n - WINDOW

        if self.is_training:
            start_idx = np.random.randint(0, max_start + 1)
        elif self.windows_per_trace == 1 or max_start == 0:
            start_idx = 0
        else:
            start_idx = int(round(k * max_start / (self.windows_per_trace - 1)))

        end_idx = start_idx + WINDOW
        csi_window = csi[start_idx:end_idx]
        ati_window = ati[start_idx:end_idx].copy()
        ati_window[-1] = 0.0  # same padding as the validation/test samples

        # complex64 -> two float32 values (real and imaginary)
        frames = np.ascontiguousarray(csi_window).view(np.float32).reshape(WINDOW, 1, 3, 1024, 2).copy()

        # ==========================================
        # DATA AUGMENTATION (training only)
        # ==========================================
        if self.is_training:
            # 1. Gaussian noise, relative to the signal scale (50% of the time)
            if np.random.rand() < 0.5:
                sigma = 0.02 * float(frames.std())
                frames = frames + np.random.normal(0, sigma, frames.shape).astype(np.float32)

            # 2. Subcarrier masking (50% of the time): zero 5% of the subcarriers
            if np.random.rand() < 0.5:
                num_mask = int(1024 * 0.05)
                mask_indices = np.random.choice(1024, num_mask, replace=False)
                frames[:, :, :, mask_indices, :] = 0.0

            # 3. Packet loss (50% of the time): a dropped frame repeats the previous one.
            #    (Zeroing frames would create huge artificial spikes once the static
            #    channel is subtracted inside the model.)
            if np.random.rand() < 0.5:
                num_drop = int(WINDOW * 0.05)
                drop_indices = np.sort(np.random.choice(WINDOW, num_drop, replace=False))
                for i in drop_indices:
                    if i > 0:
                        frames[i] = frames[i - 1]
        # ==========================================

        return {
            "frames": torch.from_numpy(frames),
            "intervals": torch.from_numpy(ati_window),
            "label": torch.tensor(sample_info["headcount"], dtype=torch.long),
        }