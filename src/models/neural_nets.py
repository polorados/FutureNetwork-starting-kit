import torch
from torch import nn

class CSICNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (Batch * Time, 3 Rx Antennas, 1024 Subcarriers)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Compress the remaining subcarriers to 1 value per channel
            nn.Dropout(p=0.4)       # Heavy dropout to prevent memorizing room signatures
        )

    def forward(self, x):
        # x shape: (B * 600, 3, 1024)
        features = self.cnn(x)      # Output shape: (B * 600, 64, 1)
        return features.squeeze(-1) # Output shape: (B * 600, 64)


class AdvancedCSINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = CSICNNFeatureExtractor()
        
        # LSTM to process the 600 sequential frames of 64 features
        # batch_first=True expects input shape (Batch, Time, Features)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        
        # Classifier takes the LSTM output (128) + the interval average (1)
        self.classifier = nn.Sequential(
            nn.Linear(129, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 5) # 5 possible headcount classes (0 to 4)[cite: 7]
        )

    def forward(self, frames, intervals):
        has_batch = (frames.dim() == 6)
        if not has_batch:
            frames = frames.unsqueeze(0)
            intervals = intervals.unsqueeze(0)
            
        B, T, Tx, Rx, Sub, Comp = frames.shape
        
        # 1. Convert complex components to magnitude[cite: 8]
        mag = torch.norm(frames, dim=-1) # Shape: (B, 600, 1, 3, 1024)
        
        # 2. Drop the Tx dimension since it is always 1[cite: 7]
        mag = mag.squeeze(2) # Shape: (B, 600, 3, 1024)
        
        # 3. Time-Distributed CNN Processing
        # Flatten Batch and Time dimensions together to feed into the 1D CNN
        mag_flat = mag.view(B * T, Rx, Sub) # Shape: (B * 600, 3, 1024)
        
        # Extract features for every single frame
        cnn_features_flat = self.feature_extractor(mag_flat) # Shape: (B * 600, 64)
        
        # Unflatten back to a sequence of frames
        cnn_sequence = cnn_features_flat.view(B, T, 64) # Shape: (B, 600, 64)
        
        # 4. Temporal Processing with LSTM
        # lstm_out contains all hidden states, (h_n, c_n) contains the final state
        lstm_out, (h_n, c_n) = self.lstm(cnn_sequence)
        
        # We only care about the final hidden state of the top LSTM layer (h_n[-1])
        # This state contains the memory of the entire 3-second sequence
        final_temporal_feature = h_n[-1] # Shape: (B, 128)
        
        # 5. Process ATI intervals[cite: 8]
        interval_mean = intervals.mean(dim=1, keepdim=True) # Shape: (B, 1)
        
        # 6. Concatenate and classify
        combined = torch.cat((final_temporal_feature, interval_mean), dim=1) # Shape: (B, 129)
        logits = self.classifier(combined) # Shape: (B, 5)
        
        if not has_batch:
            logits = logits.squeeze(0)
            
        return logits