import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch

from src.dataset import RoomSenseDataset
from src.models.neural_nets import AdvancedCSINet

WEIGHTS_PATH = "best_model.pt"
OUTPUT_PATH = Path("model.onnx")
DATA_DIR = Path("data/labeled")   # only used for the sanity check on real samples
N_CHECK = 20                      # real samples used for the sanity check


def main(allow_untrained=False):
    # 1. Load the PyTorch model -- fail loudly if there are no trained weights
    model = AdvancedCSINet()

    if Path(WEIGHTS_PATH).exists():
        print(f"Loading trained weights from {WEIGHTS_PATH}...")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    elif allow_untrained:
        print(f"WARNING: '{WEIGHTS_PATH}' not found. Exporting an UNTRAINED model.")
    else:
        raise FileNotFoundError(
            f"'{WEIGHTS_PATH}' not found in {Path.cwd()}. Run training first (or call "
            f"main(allow_untrained=True) to test the export only). An untrained model scores ~0.20."
        )

    model.eval()  # disables dropout

    # 2. Dummy inputs with the exact evaluation shapes
    dummy_frames = torch.randn(600, 1, 3, 1024, 2, dtype=torch.float32)
    dummy_intervals = torch.ones(600, dtype=torch.float32) * 5000.0
    dummy_intervals[-1] = 0.0
    example_inputs = (dummy_frames, dummy_intervals)

    # 3. Export to ONNX
    print(f"\nExporting model to {OUTPUT_PATH}...")
    torch.onnx.export(
        model,
        example_inputs,
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["frames", "intervals"],
        output_names=["logits"],
        external_data=False,
    )

    # 4. Verify the exported model
    print("\nVerifying exported ONNX model against platform requirements...")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(OUTPUT_PATH), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    # 4a. Interface check (the platform feeds exactly these two names)
    input_names = [i.name for i in session.get_inputs()]
    output_names = [o.name for o in session.get_outputs()]
    print(f"Inputs: {input_names} | Outputs: {output_names}")
    if input_names != ["frames", "intervals"] or output_names != ["logits"]:
        raise ValueError("Export failed: ONNX inputs/outputs do not match the required interface.")

    ort_inputs = {"frames": dummy_frames.numpy(), "intervals": dummy_intervals.numpy()}
    session.run(["logits"], ort_inputs)  # warm-up

    # 4b. Timing, averaged over a few runs
    n_runs = 5
    start = perf_counter()
    for _ in range(n_runs):
        ort_outs = session.run(["logits"], ort_inputs)
    elapsed_time = (perf_counter() - start) / n_runs

    logits_out = ort_outs[0]
    print(f"Output shape: {logits_out.shape} (Expected: (5,))")
    print(f"Output type: {logits_out.dtype} (Expected: float32)")
    if logits_out.shape != (5,) or logits_out.dtype != np.float32:
        raise ValueError("Export failed: Output does not match required (5,) float32 format.")

    size_bytes = OUTPUT_PATH.stat().st_size
    print(f"Model size: {size_bytes / (1024 * 1024):.2f} MB (Limit: 100.00 MB)")
    if size_bytes >= 100_000_000:
        raise ValueError("Export failed: Model exceeds the 100 MB limit.")

    print(f"CPU prediction time: {elapsed_time:.3f} s/sample (Limit: 0.7)")
    if elapsed_time >= 0.7:
        print("WARNING: Prediction exceeds the 0.7 s/sample CPU budget! Server CPU speed may differ.")
    else:
        print("Speed check passed (on local CPU).")

    # 4c. Real-data sanity check: ONNX must reproduce PyTorch, and must not collapse to one class.
    #     NOTE: these samples come from the training data, so the accuracy printed here is NOT a
    #     generalisation estimate (use the room-held-out validation in train.py for that).
    if (DATA_DIR / "samples.csv").exists():
        print(f"\nSanity check on {N_CHECK} real samples from {DATA_DIR}...")
        ds = RoomSenseDataset(DATA_DIR, is_training=False, windows_per_trace=1)
        rng = random.Random(0)
        picks = rng.sample(range(len(ds)), min(N_CHECK, len(ds)))

        max_diff, correct = 0.0, 0
        pred_counts = Counter()
        for i in picks:
            s = ds[i]
            with torch.no_grad():
                pt = model(s["frames"], s["intervals"]).numpy()
            ox = session.run(["logits"], {"frames": s["frames"].numpy(),
                                          "intervals": s["intervals"].numpy()})[0]
            max_diff = max(max_diff, float(np.abs(pt - ox).max()))
            pred = int(np.argmax(ox))
            pred_counts[pred] += 1
            correct += int(pred == int(s["label"]))

        print(f"Max |PyTorch - ONNX| logit difference: {max_diff:.2e} (should be < 1e-3)")
        print(f"Predicted class counts: {dict(sorted(pred_counts.items()))}")
        print(f"Accuracy on these (training-room) samples: {correct / len(picks):.2f}")
        if max_diff > 1e-3:
            raise ValueError("ONNX output differs from PyTorch: the export is wrong.")
        if len(pred_counts) == 1:
            print("WARNING: every sample gets the same prediction -> the model has collapsed.")
    else:
        print(f"\n(Skipping real-data check: {DATA_DIR / 'samples.csv'} not found)")

    print("\nDone: model.onnx is exported and verified.")


if __name__ == "__main__":
    main()