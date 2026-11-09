import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from time import perf_counter

# Import the baseline model
from src.models.baselines import BaselineCNN
from src.models.neural_nets import AdvancedCSINet

def main():
    # 1. Configuration
    weights_path = "best_model.pt"
    output_path = Path("model.onnx")
    
    # 2. Load the PyTorch model
    model = AdvancedCSINet()
    
    # Load the trained weights if they exist (allows testing the export even before training)
    if Path(weights_path).exists():
        print(f"Loading trained weights from {weights_path}...")
        # map_location='cpu' ensures it loads successfully even if trained on a GPU
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"Warning: '{weights_path}' not found. Exporting untrained model.")
        
    # CRITICAL: Set to evaluation mode. This disables Dropout and locks BatchNorm layers.
    model.eval()
    
    # 3. Create dummy inputs matching the exact evaluation specifications
    # The platform requires 'frames' shape (600, 1, 3, 1024, 2) and 'intervals' shape (600,)
    # We do not need real data to export to ONNX; PyTorch just needs the shapes and datatypes to trace the operations.
    dummy_frames = torch.randn(600, 1, 3, 1024, 2, dtype=torch.float32)
    
    # Simulate intervals at roughly 200 Hz (5000 microseconds)
    dummy_intervals = torch.ones(600, dtype=torch.float32) * 5000.0 
    dummy_intervals[-1] = 0.0 # Pad the final interval with 0
    
    example_inputs = (dummy_frames, dummy_intervals)
    
    # 4. Export to ONNX
    print(f"\nExporting model to {output_path}...")
    torch.onnx.export(
        model,
        example_inputs,
        str(output_path),
        export_params=True,                  # Store the trained parameter weights inside the model file
        opset_version=18,                    # ONNX opset version compatible with most platforms
        do_constant_folding=True,            # Optimize constant operations for faster CPU inference
        input_names=["frames", "intervals"], # Strictly required by the platform[cite: 8]
        output_names=["logits"],             # Strictly required by the platform[cite: 8]
        external_data=False,
    )
    
# 5. Verify the exported model
    print("\nVerifying exported ONNX model against platform requirements...")
    options = ort.SessionOptions()
    
    # Optional: Configure threads to match the example script configuration[cite: 10]
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    
    session = ort.InferenceSession(str(output_path), sess_options=options, providers=["CPUExecutionProvider"])
    
    # Convert dummy tensors to numpy for ONNX Runtime
    ort_inputs = {
        "frames": dummy_frames.numpy(),
        "intervals": dummy_intervals.numpy()
    }
    
    # --- WARM-UP RUN ---
    # Run once to initialize memory and avoid compilation overhead skewing the time[cite: 10]
    session.run(["logits"], ort_inputs)
    
    # --- TIMED RUN ---
    start_time = perf_counter()
    ort_outs = session.run(["logits"], ort_inputs)
    elapsed_time = perf_counter() - start_time
    
    logits_out = ort_outs[0]
    
    # Check shapes and types
    print(f"Output shape: {logits_out.shape} (Expected: (5,))")
    print(f"Output type: {logits_out.dtype} (Expected: float32)")
    
    if logits_out.shape != (5,) or logits_out.dtype != np.float32:
        raise ValueError("❌ Export failed: Output does not match required (5,) float32 format.")
        
    # Check file size limits
    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB (Limit: 100.00 MB)")
    
    if size_bytes >= 100_000_000:
        raise ValueError("❌ Export failed: Model exceeds the 100 MB limit.")
        
    # Check prediction time
    print(f"CPU prediction time: {elapsed_time:.3f} s/sample")
    if elapsed_time >= 0.7:
        print("⚠️ WARNING: Prediction exceeds the 0.7 s/sample CPU budget![cite: 7]")
        print("Estimate excludes loading overhead; server CPU speed may differ.[cite: 10]")
    else:
        print("✅ Speed check passed (on local CPU).")
        
    print("\n✅ Success! model.onnx is verified and ready for submission.")

if __name__ == "__main__":
    main()