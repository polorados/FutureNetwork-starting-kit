import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path

# Import your custom dataset and model
from src.dataset import RoomSenseDataset
from src.models.baselines import BaselineCNN
from src.models.neural_nets import AdvancedCSINet

def main():
    # 1. Configuration
    data_dir = Path("data/labeled")
    batch_size = 2 # how many batches processed at a time
    epochs = 20 # combien de fois le modele va voir l'ensemble du dataset d'entrainement. 
    learning_rate = 1e-3
    
    # Automatically use GPU if available (Apple Silicon MPS or Nvidia CUDA), else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading & Splitting
    # Load the full labeled dataset
    full_dataset = RoomSenseDataset(data_dir, is_training=True)
    
    # Split into 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model, Loss, and Optimizer Setup
    model = AdvancedCSINet().to(device)
    
    # CrossEntropyLoss is standard for multi-class classification tasks
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_accuracy = 0.0

    # 4. The Training Loop
    for epoch in range(epochs):
        model.train() # Set model to training mode (enables dropout/batchnorm if used)
        running_loss = 0.0
        
        for batch in train_loader:
            # Move data to the correct device (CPU/GPU)
            frames = batch["frames"].to(device)
            intervals = batch["intervals"].to(device)
            labels = batch["label"].to(device)
            
            # Step A: Zero the gradients (clear old calculations)
            optimizer.zero_grad()
            
            # Step B: Forward pass (generate predictions)
            logits = model(frames, intervals)
            
            # Step C: Calculate the loss (how wrong the predictions are)
            loss = criterion(logits, labels)
            
            # Step D: Backward pass (calculate the gradients)
            loss.backward()
            
            # Step E: Update model weights
            optimizer.step()
            
            running_loss += loss.item() * frames.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # 5. Validation Loop
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        
        # Disable gradient calculation for validation to save memory and compute
        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                intervals = batch["intervals"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(frames, intervals)
                
                # The predicted headcount is the index with the highest logit score[cite: 8]
                predictions = torch.argmax(logits, dim=1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        # Submissions are scored using classification accuracy
        val_accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
        
        # 6. Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> New best model saved!")

if __name__ == "__main__":
    main()