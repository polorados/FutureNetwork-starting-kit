import torch
from torch import nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # out_channels : nb de pattern / nb de filtre utilisés
            # stride on skip une donnée donc on down sample
            # in_channels : pour chaque pixel, combien de valeur associée
            
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

        # Input: (Batch, 16, 256)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Input: (Batch, 32, 64)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Pool down to exactly 1 value per channel regardless of sequence length
            nn.AdaptiveAvgPool1d(1) 
        )
        
        # The CNN outputs 64 features + 1 feature for the interval mean = 65 features
        self.classifier = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            # 5 possible headcount classes (0, 1, 2, 3, 4)
            nn.Linear(32, 5) 
        )

    def forward(self, frames, intervals):
        # ONNX platform inputs do NOT have a batch dimension: (600, 1, 3, 1024, 2)
        # PyTorch DataLoader adds a batch dimension during training: (B, 600, 1, 3, 1024, 2)
        # This check dynamically handles both scenarios.

        #check si on est dans une situation d'évaluation ou d'entrainement, 
        # parcequ'en entrainement la donnée est ajoutée un batch à la fois donc dim = 5
        has_batch = (frames.dim() == 6)
        # ajoute une dimension de batch si elle n'existe pas.
        if not has_batch:
            frames = frames.unsqueeze(0)
            intervals = intervals.unsqueeze(0)
            
        # 1. Convert complex components to magnitude
        # The original complex-valued CSI is represented by two real-valued components in the final dimension
        # torch.norm computes the Euclidean magnitude: sqrt(real^2 + imag^2)
        mag = torch.norm(frames, dim=-1) # Shape becomes: (B, 600, 1, 3, 1024)
        
        # 2. Average across the time (600) and Tx antenna (1) dimensions
        # This simplifies the problem to static spatial/frequency features for the baseline
        # On donne ce qu'attend un CNN. des channels (comme pour le RGB) et une sequence length.
        x = mag.mean(dim=(1, 2)) # Shape becomes: (B, 3, 1024)
        
        # 3. Extract features using the 1D CNN
        cnn_features = self.feature_extractor(x) # Shape: (B, 64, 1)
        cnn_features = cnn_features.squeeze(-1)  # Shape: (B, 64)
        
        # 4. Process ATI intervals
        # intervals[i] is the elapsed time between frame i and frame i+1
        interval_mean = intervals.mean(dim=1, keepdim=True) # Shape: (B, 1)
        
        # 5. Concatenate and classify
        combined = torch.cat((cnn_features, interval_mean), dim=1) # Shape: (B, 65)
        logits = self.classifier(combined) # Classification logits corresponding to occupancy classes 0-4[cite: 8]
        
        # Remove batch dimension if it wasn't there originally to match the strict (5,) output requirement[cite: 8]
        if not has_batch:
            logits = logits.squeeze(0)
            
        return logits