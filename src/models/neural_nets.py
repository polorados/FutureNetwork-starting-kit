import torch
from torch import nn

N_BANDS = 16        # subcarrier bands used by the temporal-statistics branch (1024 / 16 = 64 each)
DEV_SCALE = 10.0    # fixed scale so the "deviation from static channel" signal is ~O(1)
STAT_SCALE = 10.0


class CSICNNFeatureExtractor(nn.Module):
    """Per-frame 1D CNN over subcarriers.

    No BatchNorm / per-frame normalisation on purpose:
      - BatchNorm statistics differ between train (batch of 2 traces) and eval (running
        averages), which breaks on unseen rooms.
      - Per-frame norm layers (GroupNorm/InstanceNorm) would erase the amplitude of the
        temporal variation, which is exactly the occupancy signal.
    The input is already normalised in AdvancedCSINet._preprocess.
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.out_dim = out_dim  # 64 (avg) + 64 (max)

    def forward(self, x):
        # x: (B*T, 3, 1024)
        h = self.cnn(x)                                   # (B*T, 64, L)
        feat = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)  # (B*T, 128)
        return self.dropout(feat)


class AdvancedCSINet(nn.Module):
    def __init__(self, use_intervals=False):
        """use_intervals: feed ATI statistics to the classifier.
        Default False: the intervals depend on the room's Wi-Fi traffic and are an easy
        shortcut to overfit. Try True as a separate submission to compare on the leaderboard.
        """
        super().__init__()
        self.use_intervals = use_intervals
        self.feature_extractor = CSICNNFeatureExtractor()

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.3)

        # Branch with explicit temporal statistics (std + mean abs frame-to-frame change,
        # per antenna and per subcarrier band): 2 * 3 * 16 = 96 values
        self.stats_mlp = nn.Sequential(
            nn.Linear(2 * 3 * N_BANDS, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        in_dim = 128 + 64 + (2 if use_intervals else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5),
        )

    @staticmethod
    def _preprocess(frames):
        """(B, T, 1, 3, 1024, 2) -> (B, T, 3, 1024) deviation from the static channel."""
        re = frames[..., 0]
        im = frames[..., 1]
        mag = torch.sqrt(re * re + im * im + 1e-12).squeeze(2)        # (B, T, 3, 1024)

        # 1. Remove gain / AGC jitter: normalise each frame and antenna by its mean magnitude
        mag = mag / (mag.mean(dim=3, keepdim=True) + 1e-6)

        # 2. Remove the static background (room signature): subtract the temporal mean
        dev = mag - mag.mean(dim=1, keepdim=True)
        return dev

    @staticmethod
    def _temporal_stats(dev):
        B, T, R, S = dev.shape
        std = torch.sqrt((dev * dev).mean(dim=1) + 1e-8)               # (B, 3, 1024)
        diff = (dev[:, 1:] - dev[:, :-1]).abs().mean(dim=1)            # (B, 3, 1024)
        s = torch.cat([std, diff], dim=1)                              # (B, 6, 1024)
        s = s.reshape(B, 2 * R, N_BANDS, S // N_BANDS).mean(dim=-1)    # (B, 6, 16)
        return s.reshape(B, -1) * STAT_SCALE                           # (B, 96)

    def forward(self, frames, intervals):
        has_batch = (frames.dim() == 6)
        if not has_batch:
            frames = frames.unsqueeze(0)
            intervals = intervals.unsqueeze(0)

        B, T, Tx, Rx, Sub, Comp = frames.shape

        dev = self._preprocess(frames)                                 # (B, T, 3, 1024)

        # Time-distributed CNN
        cnn_in = (dev * DEV_SCALE).reshape(B * T, Rx, Sub)
        cnn_seq = self.feature_extractor(cnn_in).reshape(B, T, -1)     # (B, T, 128)

        # Temporal modelling: mean over all hidden states is more stable than only the last one
        lstm_out, _ = self.lstm(cnn_seq)
        temporal_feat = lstm_out.mean(dim=1)                           # (B, 128)

        stats_feat = self.stats_mlp(self._temporal_stats(dev))         # (B, 64)

        combined = torch.cat([temporal_feat, stats_feat], dim=1)

        if self.use_intervals:
            iv = intervals[:, :-1]                                     # last value is padding (0)
            m = iv.mean(dim=1, keepdim=True)
            sd = torch.sqrt(((iv - m) ** 2).mean(dim=1) + 1e-6)
            extra = torch.stack([m.squeeze(1) / 5000.0, sd / 5000.0], dim=1).clamp(max=20.0)
            combined = torch.cat([combined, extra], dim=1)
        else:
            # Keep `intervals` connected to the graph so the ONNX file still exposes the
            # required "intervals" input (unused inputs can be dropped by the exporter).
            combined = combined + 0.0 * intervals.sum()

        logits = self.classifier(combined)                             # (B, 5)

        if not has_batch:
            logits = logits.squeeze(0)
        return logits