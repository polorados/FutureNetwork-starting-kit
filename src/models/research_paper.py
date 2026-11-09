import torch
from torch import nn
from torch.autograd import Function

# 1. Gradient Reversal Layer (GRL)
# The GRL reverses gradients during backpropagation to force the CNN backbone 
# to produce domain-invariant features.
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GRL(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# 2. WiCount-DASL Architecture
class WiCountDASL(nn.Module):
    def __init__(self, num_classes=5, num_domains=3):
        super().__init__()
        
        # Shared CNN Backbone to extract compact representation f[cite: 17]
        self.shared_backbone = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(p=0.4)
        )
        
        # Label Classifier (FC_cls)[cite: 17]
        # Maps the extracted feature to the class space
        self.label_classifier = nn.Sequential(
            nn.Linear(65, 32), # 64 CNN features + 1 interval feature
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # Domain Classifier (FC_dom)
        # Predicts the scenario label based on the shared feature representation
        self.grl = GRL(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains) # e.g., predicting which room the data came from
        )

    def forward(self, frames, intervals, alpha=1.0, return_domain=False):
        has_batch = (frames.dim() == 6)
        if not has_batch:
            frames = frames.unsqueeze(0)
            intervals = intervals.unsqueeze(0)
            
        # Convert to amplitude-time representations[cite: 16]
        mag = torch.norm(frames, dim=-1) 
        mag = mag.squeeze(2) # Drop Tx antenna dimension
        x = mag.mean(dim=1)  # Average over time for the CNN (or use LSTM setup here)
        
        # Extract Shared Features
        f_cnn = self.shared_backbone(x).squeeze(-1)
        interval_mean = intervals.mean(dim=1, keepdim=True)
        f = torch.cat((f_cnn, interval_mean), dim=1)
        
        # Label Classification Branch[cite: 17]
        class_logits = self.label_classifier(f)
        
        if not has_batch:
            class_logits = class_logits.squeeze(0)
            
        # During evaluation/ONNX export, we only need the class predictions
        if not return_domain:
            return class_logits
            
        # Domain-Adversarial Branch
        self.grl.alpha = alpha
        f_reversed = self.grl(f)
        domain_logits = self.domain_classifier(f_reversed)
        
        return class_logits, domain_logits