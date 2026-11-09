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
    batch_size = 8
    epochs = 100
    warm_up_epochs = 50 
    pseudo_label_warmup = 15
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    labeled_dataset = ResearchLabeledDataset(labeled_dir)
    unlabeled_dataset = ResearchUnlabeledDataset(unlabeled_dir, domain_map=labeled_dataset.domain_map)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
    
    num_domains = max(labeled_dataset.domain_map.values()) + 1

    model = WiCountDASL(num_classes=5, num_domains=num_domains).to(device)
    
    class_weights = torch.tensor([1.0, 0.7, 3.0, 3.0, 3.0], device=device)
    
    # 1. Perte AVEC poids pour les données dont on connaît la vraie réponse
    criterion_sup = nn.CrossEntropyLoss(weight=class_weights)
    
    # 2. Perte SANS poids pour les pseudo-labels (laisse la confiance naturelle du modèle agir)
    criterion_unsup = nn.CrossEntropyLoss()
    criterion_dom = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    thresholds = torch.ones(5, device=device) * 0.95

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # --- CORRECTION 4b : Poids progressif/différé pour la perte non supervisée ---
        unsup_weight = 0.0 if epoch < pseudo_label_warmup else 1.0
        # -----------------------------------------------------------------------------
        
        if epoch < warm_up_epochs:
            lambda_dann = 0.0
        else:
            p = float(epoch - warm_up_epochs) / (epochs - warm_up_epochs)
            lambda_dann = 2. / (1. + np.exp(-10 * p)) - 1
            
        for labeled_batch, unlabeled_batch in zip(cycle(labeled_loader), unlabeled_loader):
            # 1. Supervised Branch
            frames_l = labeled_batch["frames"].to(device)
            intervals_l = labeled_batch["intervals"].to(device)
            labels_l = labeled_batch["label"].to(device)
            domain_l = labeled_batch["domain"].to(device)
            
            optimizer.zero_grad()
            
            logits_l, dom_logits_l = model(frames_l, intervals_l, alpha=lambda_dann, return_domain=True)
            loss_sup = criterion_sup(logits_l, labels_l)
            
            # 2. Unsupervised SSL Branch
            frames_w = unlabeled_batch["frames_w"].to(device)
            frames_s = unlabeled_batch["frames_s"].to(device)
            intervals_u = unlabeled_batch["intervals"].to(device)
            domain_u = unlabeled_batch["domain"].to(device)
            
            # --- FIX 3: Lock dropout to generate stable pseudo-labels ---
            model.eval()
            with torch.no_grad():
                logits_w = model(frames_w, intervals_u, return_domain=False)
                probs_w = torch.softmax(logits_w, dim=1)
                max_probs, pseudo_labels = torch.max(probs_w, dim=1)
            model.train()
            # ------------------------------------------------------------
            
            # --- FIX 2: Update thresholds based on the full current batch ---
            with torch.no_grad():
                for c in range(5):
                    c_mask = (pseudo_labels == c)
                    if c_mask.sum() > 0:
                        avg_conf = max_probs[c_mask].mean().item()
                        thresholds[c] = 0.99 * thresholds[c].item() + 0.01 * avg_conf
            # ----------------------------------------------------------------
            
            mask = max_probs >= thresholds[pseudo_labels]
            
            # Always pass the strongly augmented frames to generate the domain logits
            logits_s, dom_logits_u = model(frames_s, intervals_u, alpha=lambda_dann, return_domain=True)
            
            loss_unsup = torch.tensor(0.0, device=device)
            if mask.sum() > 0:
                loss_unsup = criterion_unsup(logits_s[mask], pseudo_labels[mask])

            # 3. Domain Adversarial Branch
            loss_dom = torch.tensor(0.0, device=device)
            if lambda_dann > 0:
                dom_logits_combined = torch.cat([dom_logits_l, dom_logits_u], dim=0)
                dom_labels_combined = torch.cat([domain_l, domain_u], dim=0)
                loss_dom = criterion_dom(dom_logits_combined, dom_labels_combined)
            
            total_loss = loss_sup + (unsup_weight * loss_unsup) + (1.0 * loss_dom)
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        epoch_loss = running_loss / len(unlabeled_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | DANN W: {lambda_dann:.2f} | Thresh: {thresholds.mean().item():.2f}")
        
        # --- FIX 1: Save at the end of training ---
        # The loss intentionally spikes when DANN kicks in, so don't track best_loss here.
        if epoch == epochs - 1:
            torch.save(model.state_dict(), "best_research_model.pt")
            print("  -> Final DANN-trained model saved!")
            
if __name__ == "__main__":
    main()