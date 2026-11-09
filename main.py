import argparse

# Standard pipeline
from src.train import main as train_standard
from src.export import main as export_standard

# Research pipeline
from src.research_train import main as train_research
from src.research_export import main as export_research

def main():
    parser = argparse.ArgumentParser(description="RoomSense Competition CLI")
    
    parser.add_argument(
        "--mode", type=str, choices=["train", "export", "all"], required=True,
        help="Action to perform."
    )
    parser.add_argument(
        "--pipeline", type=str, choices=["standard", "research"], default="standard",
        help="Choose between the standard models (baselines/neural_nets) or the DASL research paper model."
    )
    
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        print("=" * 50)
        print(f"🚀 STARTING {args.pipeline.upper()} TRAINING PHASE...")
        print("=" * 50)
        if args.pipeline == "standard":
            train_standard()
        else:
            train_research()

    if args.mode in ["export", "all"]:
        print("\n" + "=" * 50)
        print(f"📦 STARTING {args.pipeline.upper()} ONNX EXPORT PHASE...")
        print("=" * 50)
        if args.pipeline == "standard":
            export_standard()
        else:
            export_research()

if __name__ == "__main__":
    main()