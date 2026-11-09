import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import cycle
import numpy as np

from src.research_dataset import ResearchLabeledDataset, ResearchUnlabeledDataset
from src.models.research_paper import WiCountDASL

def main():
    labeled_dir = Path("data/labeled")
    unlabeled_dir = Path("data/unlabeled")
    batch_size = 2
    epochs = 100
    warm_up_epochs = 50 # Start DANN after epoch 50[cite: 19]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labeled_dataset = ResearchLabeledDataset(labeled_dir)
    unlabeled_dataset = ResearchUnlabeledDataset(unlabeled_dir, domain_map=labeled_dataset.domain_map)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    num_domains = max(labeled_dataset.domain_map.values()) + 1

    model = WiCountDASL(num_classes=5, num_domains=num_domains).to(device)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Initialize adaptive classwise thresholds[cite: 19]
    thresholds = torch.ones(5, device=device) * 0.95
    
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Calculate dynamic DANN weight[cite: 19]
        if epoch < warm_up_epochs:
            lambda_dann = 0.0
        else:
            p = float(epoch - warm_up_epochs) / (epochs - warm_up_epochs)
            lambda_dann = 2. / (1. + np.exp(-10 * p)) - 1 # Standard DANN scheduling
            
        # Zip the loaders. Cycle the labeled one since unlabeled is usually much larger.
        for labeled_batch, unlabeled_batch in zip(cycle(labeled_loader), unlabeled_loader):
            # 1. Supervised Branch
            frames_l = labeled_batch["frames"].to(device)
            intervals_l = labeled_batch["intervals"].to(device)
            labels_l = labeled_batch["label"].to(device)
            domain_l = labeled_batch["domain"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass for labeled data[cite: 18]
            logits_l, dom_logits_l = model(frames_l, intervals_l, alpha=lambda_dann, return_domain=True)
            loss_sup = criterion_cls(logits_l, labels_l)
            
            # 2. Unsupervised SSL Branch[cite: 17]
            frames_w = unlabeled_batch["frames_w"].to(device)
            frames_s = unlabeled_batch["frames_s"].to(device)
            intervals_u = unlabeled_batch["intervals"].to(device)
            domain_u = unlabeled_batch["domain"].to(device)
            
            # Get weak pseudolabels without tracking gradients
            with torch.no_grad():
                logits_w = model(frames_w, intervals_u, return_domain=False)
                probs_w = torch.softmax(logits_w, dim=1)
                max_probs, pseudo_labels = torch.max(probs_w, dim=1)
                
            # Filter by classwise adaptive thresholds[cite: 18]
            mask = max_probs >= thresholds[pseudo_labels]
            
            loss_unsup = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                logits_s, dom_logits_u = model(frames_s, intervals_u, alpha=lambda_dann, return_domain=True)
                loss_unsup = criterion_cls(logits_s[mask], pseudo_labels[mask])
                
                # Update thresholds with exponential moving average
                for c in range(5):
                    c_mask = (pseudo_labels == c) & mask
                    if c_mask.sum() > 0:
                        avg_conf = max_probs[c_mask].mean().item()
                        thresholds[c] = 0.99 * thresholds[c].item() + 0.01 * avg_conf

            # 3. Domain Adversarial Branch[cite: 18]
            loss_dom = torch.tensor(0.0, device=device)
            if lambda_dann > 0:
                # We already calculated dom_logits_l. Let's calculate dom_logits_u if we skipped it above.
                if mask.sum() == 0:
                    _, dom_logits_u = model(frames_s, intervals_u, alpha=lambda_dann, return_domain=True)
                
                dom_logits_combined = torch.cat([dom_logits_l, dom_logits_u], dim=0)
                dom_labels_combined = torch.cat([domain_l, domain_u], dim=0)
                loss_dom = criterion_dom(dom_logits_combined, dom_labels_combined)
            
            # Total Composite Loss[cite: 18]
            total_loss = loss_sup + (1.0 * loss_unsup) + (1.0 * loss_dom)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        epoch_loss = running_loss / len(unlabeled_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | DANN W: {lambda_dann:.2f} | Thresh: {thresholds.mean().item():.2f}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_research_model.pt")
            
if __name__ == "__main__":
    main()