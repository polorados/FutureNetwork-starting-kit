import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from time import perf_counter

from src.models.research_paper import WiCountDASL

def main():
    weights_path = "best_research_model.pt"
    output_path = Path("research_model.onnx")
    
    # --- DOMAIN MISMATCH FIX: Set num_domains to match your training set ---
    # Update '6' below if your labeled dataset actually mapped to a different number of layouts
    model = WiCountDASL(num_classes=5, num_domains=6) 
    
    if Path(weights_path).exists():
        print(f"Loading trained weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("domain_classifier")}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        print(f"Warning: '{weights_path}' not found. Exporting untrained model.")
        
    model.eval()
    
    dummy_frames = torch.randn(600, 1, 3, 1024, 2, dtype=torch.float32)
    dummy_intervals = torch.ones(600, dtype=torch.float32) * 5000.0 
    dummy_intervals[-1] = 0.0
    
    example_inputs = (dummy_frames, dummy_intervals)
    
    # --- FIX 5: Declare Dynamic Axes ---
    # This explicitly tells ONNX that the batch size (index 0) can change dynamically
    dynamic_axes = {
        "frames": {0: "batch_size"},
        "intervals": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
    # -----------------------------------
    
    print(f"\nExporting model to {output_path}...")
    torch.onnx.export(
        model,
        example_inputs,
        str(output_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["frames", "intervals"],
        output_names=["logits"],
        external_data=False, 
    )
    
    print("\nVerifying exported ONNX model...")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(output_path), sess_options=options, providers=["CPUExecutionProvider"])
    
    ort_inputs = {"frames": dummy_frames.numpy(), "intervals": dummy_intervals.numpy()}
    session.run(["logits"], ort_inputs) 
    
    start_time = perf_counter()
    ort_outs = session.run(["logits"], ort_inputs)
    elapsed_time = perf_counter() - start_time
    
    logits_out = ort_outs[0]
    
    # Check shape[-1] instead of strictly (5,) to accommodate the dynamic batch dimension
    if logits_out.shape[-1] != 5 or logits_out.dtype != np.float32:
        raise ValueError(f"❌ Export failed: Output format incorrect. Got shape: {logits_out.shape}")
        
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB (Limit: 100.00 MB)")
    if size_mb >= 100:
        raise ValueError("❌ Export failed: Model exceeds limit.")
        
    print(f"CPU prediction time: {elapsed_time:.3f} s/sample")
    print("✅ Success! research_model.onnx is ready.")

if __name__ == "__main__":
    main()