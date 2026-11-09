"""Train the baseline classifier with PyTorch on CUDA and export it to ONNX.

Keep this script beside 1er_essai.py, which provides the shared data and ONNX
helpers. Run from this folder:
    python 1er_essai_pytorch.py --data-dir labeled/ --output model_pytorch.onnx
"""

import argparse
import importlib
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


baseline = importlib.import_module("1er_essai")


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_scale = np.maximum(features.std(axis=0).astype(np.float32), 1e-6)
    normalized = (features - feature_mean) / feature_scale

    class_counts = np.bincount(labels, minlength=baseline.NUM_CLASSES)
    class_weights = np.zeros(baseline.NUM_CLASSES, dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = len(labels) / (
        int(present.sum()) * class_counts[present]
    )
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.sum()

    inputs = torch.from_numpy(normalized).to(device)
    targets = torch.from_numpy(labels).to(device)
    loss_weights = torch.from_numpy(sample_weights).to(device)
    model = nn.Linear(baseline.NUM_FEATURES, baseline.NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = (per_sample_loss * loss_weights).sum()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch + 1 == epochs:
            with torch.no_grad():
                accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
            print(
                f"Epoch {epoch + 1:>3}/{epochs}: "
                f"weighted loss={loss.item():.4f}, "
                f"training window accuracy={accuracy:.1%}"
            )

    weights = model.weight.detach().cpu().numpy().copy()
    bias = model.bias.detach().cpu().numpy().copy()
    return weights, bias, feature_mean, feature_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "labeled",
        help="Folder containing samples.csv and the labeled CSI/ATI files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_pytorch.onnx"),
        help="Destination ONNX model path",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Training device; CUDA is required by default",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not args.data_dir.is_dir():
        parser.error(f"Training data folder not found: {args.data_dir}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA GPU. "
            f"PyTorch version: {torch.__version__}; "
            f"PyTorch CUDA build: {torch.version.cuda!r}. "
            "Check that WSL can see the NVIDIA GPU (`nvidia-smi`) and that "
            "this Conda environment has a CUDA-enabled PyTorch build. "
            "Use --device cpu only if you explicitly want CPU training."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        print(f"Training device: CUDA - {torch.cuda.get_device_name(device)}")
    else:
        print("Training device: CPU (explicitly requested)")

    features, labels, frames, intervals = baseline.load_training_windows(
        args.data_dir
    )
    print(
        f"Loaded {len(features)} windows from labeled traces; "
        f"class counts (windows): "
        f"{np.bincount(labels, minlength=baseline.NUM_CLASSES).tolist()}"
    )
    print(
        "Training accuracy is only a sanity check; it is not a held-out "
        "estimate of generalization to unseen rooms."
    )
    weights, bias, feature_mean, feature_scale = train_model(
        features, labels, args.epochs, args.seed, device
    )
    baseline.export_and_check(
        weights,
        bias,
        feature_mean,
        feature_scale,
        frames,
        intervals,
        args.output,
    )


if __name__ == "__main__":
    main()
