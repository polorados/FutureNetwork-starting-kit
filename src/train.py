import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import RoomSenseDataset
from src.models.neural_nets import AdvancedCSINet

# ---------------- Configuration ----------------
DATA_DIR = Path("data/labeled")
BATCH_SIZE = 8          # lower to 4 if you run out of memory
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
TRAIN_REPEATS = 2       # random windows drawn per training trace in each epoch
VAL_WINDOWS = 4         # fixed windows per validation trace
VAL_FRACTION = 0.2
SPLIT_BY = "room"       # "room" or "layout": what is held out for validation
SEED = 0
NUM_WORKERS = 0         # keep 0 (the eval-time trace cache lives in the main process)
WEIGHTS_PATH = "best_model.pt"
# -----------------------------------------------


def group_key(sample, split_by):
    """Traces from the same room (or layout) must all go on the same side of the split.
    'unknown' metadata means we cannot tell: treat each such trace as its own group."""
    layout, room = sample["layout"], sample["room"]
    if layout == "unknown" or room == "unknown":
        return f"unknown_{sample['identifier']}"
    return layout if split_by == "layout" else f"{layout}_{room}"


def split_by_group(samples, val_fraction, split_by, rng):
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        groups[group_key(s, split_by)].append(i)
    if len(groups) < 2:
        raise RuntimeError("Need at least 2 distinct rooms/layouts to build a held-out validation set.")

    target = max(1, int(round(val_fraction * len(samples))))
    keys = list(groups)

    for _ in range(200):  # retry until the split is usable
        rng.shuffle(keys)
        val = []
        for k in keys:
            if len(val) >= target:
                break
            val += groups[k]
        val_set = set(val)
        train = [i for i in range(len(samples)) if i not in val_set]
        val_labels = {samples[i]["headcount"] for i in val}
        train_labels = {samples[i]["headcount"] for i in train}
        if len(val_labels) >= 2 and val_labels <= train_labels:
            return train, val
    raise RuntimeError("Could not find a room-level split with enough class diversity; "
                       "try SPLIT_BY='room' or a different SEED.")


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    pred_counts = Counter()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            intervals = batch["intervals"].to(device)
            labels = batch["label"].to(device)
            logits = model(frames, intervals)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            pred_counts.update(preds.tolist())
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total, loss_sum / total, pred_counts


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data: split by ROOM, not by random trace ----
    full_train_dataset = RoomSenseDataset(DATA_DIR, is_training=True)
    full_val_dataset = RoomSenseDataset(DATA_DIR, is_training=False, windows_per_trace=VAL_WINDOWS)
    samples = full_train_dataset.samples

    train_traces, val_traces = split_by_group(samples, VAL_FRACTION, SPLIT_BY, rng)

    train_rooms = {group_key(samples[i], SPLIT_BY) for i in train_traces}
    val_rooms = {group_key(samples[i], SPLIT_BY) for i in val_traces}
    assert not (train_rooms & val_rooms), "Room leakage between train and validation!"
    print(f"Split by {SPLIT_BY}: {len(train_traces)} train traces ({len(train_rooms)} groups) | "
          f"{len(val_traces)} val traces ({len(val_rooms)} groups)")
    print("Train label counts:", dict(sorted(Counter(samples[i]['headcount'] for i in train_traces).items())))
    print("Val   label counts:", dict(sorted(Counter(samples[i]['headcount'] for i in val_traces).items())))

    K = full_val_dataset.windows_per_trace
    val_indices = [t * K + k for t in val_traces for k in range(K)]

    train_dataset = Subset(full_train_dataset, train_traces * TRAIN_REPEATS)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # ---- Model / loss / optimiser ----
    model = AdvancedCSINet().to(device)
    class_weights = torch.tensor([1.0, 0.7, 3.0, 3.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, best_val_loss = -1.0, float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            frames = batch["frames"].to(device)
            intervals = batch["intervals"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(frames, intervals)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # LSTM over 600 steps
            optimizer.step()

            running_loss += loss.item() * frames.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)
        val_acc, val_loss, pred_counts = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc (unseen rooms): {val_acc:.4f}")
        print(f"  predicted classes on val: {dict(sorted(pred_counts.items()))}")
        if len(pred_counts) == 1:
            print("  WARNING: the model predicts a single class -> collapsed, accuracy is just class frequency")

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print("  -> New best model saved!")

    print(f"\nBest validation accuracy on held-out {SPLIT_BY}s: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()