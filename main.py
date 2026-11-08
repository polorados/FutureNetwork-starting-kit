import argparse

# Import the main functions from your source scripts
from src.train import main as train_model
from src.export import main as export_model

def main():
    parser = argparse.ArgumentParser(
        description="RoomSense Competition: Headcount Estimation from CSI"
    )
    
    # Add a required argument to specify what the user wants to do
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "export", "all"],
        required=True,
        help="Action to perform: 'train' to train the model, 'export' to convert to ONNX, or 'all' to do both sequentially."
    )
    
    args = parser.parse_args()

    # Route the execution based on the user's choice
    if args.mode in ["train", "all"]:
        print("=" * 50)
        print("🚀 STARTING TRAINING PHASE...")
        print("=" * 50)
        train_model()
        print("\n✅ Training complete.")

    if args.mode in ["export", "all"]:
        print("\n" + "=" * 50)
        print("📦 STARTING ONNX EXPORT PHASE...")
        print("=" * 50)
        export_model()

if __name__ == "__main__":
    main()