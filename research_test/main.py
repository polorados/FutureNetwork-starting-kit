import torch
import torch.nn as nn
import torch.nn.functional as functional


class GradientReversalLayer(torch.autograd.Function):
    """Identity in the forward pass; multiplies the gradient by -1 in the backward pass.

    Placed between the shared feature extractor and the domain classifier, it makes the
    feature extractor *maximize* the domain classifier's loss (adversarial alignment, Eq. 10).
    """

    @staticmethod
    def forward(context, features):
        return features.view_as(features)

    @staticmethod
    def backward(context, incoming_gradient):
        return -incoming_gradient


class WiCountNetwork(nn.Module):
    """Shared CNN backbone (Eq. 7) with two heads: a label classifier (Eq. 8) and a
    domain classifier behind a gradient reversal layer (Eq. 10).

    Input : CSI amplitude matrix of shape (batch, time_steps N, subcarriers S).
    Output: (class logits, domain logits).
    """

    def __init__(self, num_classes, num_domains, feature_dimension=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, feature_dimension), nn.ReLU(), nn.Dropout(0.6),
        )
        self.label_classifier = nn.Linear(feature_dimension, num_classes)
        self.domain_classifier = nn.Linear(feature_dimension, num_domains)

    def forward(self, csi_amplitude):
        features = self.backbone(csi_amplitude.unsqueeze(1))  # add a channel dimension
        class_logits = self.label_classifier(features)
        domain_logits = self.domain_classifier(GradientReversalLayer.apply(features))
        return class_logits, domain_logits


def apply_strong_augmentation(csi_amplitude, noise_std=0.1, subcarrier_drop_probability=0.1):
    """Targeted signal-level strong augmentation, Eq. (6):
    x_strong = (x + sigma * N(0, 1)) ⊙ m · s

    - Random Channel Noise injection : Gaussian noise added to the amplitude.
    - Random Subcarrier Masking      : a binary mask removes a random subset of subcarriers.
    - Random Gain Scaling            : one factor drawn from U(0.9, 1.1) per sample.
    Parameters are re-drawn at every call, as described in the paper.
    """
    batch_size, _, num_subcarriers = csi_amplitude.shape
    subcarrier_mask = (torch.rand(batch_size, 1, num_subcarriers) > subcarrier_drop_probability).float()
    gain_scale = torch.empty(batch_size, 1, 1).uniform_(0.9, 1.1)
    noisy_amplitude = csi_amplitude + noise_std * torch.randn_like(csi_amplitude)
    return noisy_amplitude * subcarrier_mask * gain_scale


def train_wicount_dasl(labeled_loader, unlabeled_loader, num_classes, num_domains,
                       total_epochs=100, warmup_epochs=50,
                       unsupervised_weight=1.0, domain_adversarial_weight=1.0,
                       threshold_momentum=0.9):
    """Algorithm 1: WiCount-DASL model training. Returns the trained model.

    labeled_loader   yields (samples, class_labels, domain_labels)
    unlabeled_loader yields (samples, domain_labels)
    """
    # Require: Labeled data loader Dl, unlabeled data loader Du
    # Require: Total epochs T, warm-up epoch Twarm, weights λu, λdann, initial thresholds τc(0) = 0.95
    # Ensure: Trained model parameters θ

    # 1: Initialize network parameters and optimizer;
    model = WiCountNetwork(num_classes, num_domains)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # 2: Initialize classwise thresholds {τc} with τc(0);
    class_thresholds = torch.full((num_classes,), 0.95)

    # 3: for t = 1 to T do
    for epoch in range(1, total_epochs + 1):
        # 4: for each training iteration do
        # 5: Sample labeled batch (xl, yl, dl) from Dl;
        # 6: Sample unlabeled batch (xu, du) from Du;
        for (labeled_samples, labeled_targets, labeled_domains), \
                (unlabeled_samples, unlabeled_domains) in zip(labeled_loader, unlabeled_loader):

            # 7: Forward xl through the model to obtain logits ŷl;
            # 10: Obtain predictions ŷw = p(f(xw));   (weak augmentation is the identity, Sec. IV-A)
            # (labeled and weakly augmented unlabeled data go through the model together, so that
            #  batch-norm statistics cover both, as in Sec. IV-E)
            num_labeled = len(labeled_samples)
            all_class_logits, all_domain_logits = model(torch.cat([labeled_samples, unlabeled_samples]))
            labeled_logits = all_class_logits[:num_labeled]
            weak_probabilities = functional.softmax(all_class_logits[num_labeled:], dim=1)

            # 8: Compute supervised loss Lsup using (13);
            supervised_loss = functional.cross_entropy(labeled_logits, labeled_targets)

            # 9: Generate weakly augmented xw and strongly augmented xs from xu;
            weak_samples = unlabeled_samples
            strong_samples = apply_strong_augmentation(unlabeled_samples)

            # 11: Derive pseudolabel ŷu using (11);
            confidence, pseudo_labels = weak_probabilities.max(dim=1)

            # 12: Build mask M using the adaptive rule in (12);
            selection_mask = confidence.detach() >= class_thresholds[pseudo_labels]

            # 13: if ΣM > 0 then
            if selection_mask.sum() > 0:
                # 14: Obtain predictions ŷs = p(f(xs));
                strong_logits, _ = model(strong_samples)

                # 15: Compute consistency loss Lunsup using (14);
                per_sample_loss = functional.cross_entropy(strong_logits, pseudo_labels.detach(), reduction="none")
                unsupervised_loss = (per_sample_loss * selection_mask).mean()

                # 16: Update thresholds τc using a moving average of the selected confidences;
                for class_index in pseudo_labels[selection_mask].unique():
                    selected_in_class = selection_mask & (pseudo_labels == class_index)
                    class_thresholds[class_index] = (
                        threshold_momentum * class_thresholds[class_index]
                        + (1 - threshold_momentum) * confidence[selected_in_class].mean().detach()
                    )
            # 17: else
            else:
                # 18: Lunsup ← 0
                unsupervised_loss = torch.zeros(())
            # 19: end if

            # 20: if t ≤ Twarm then
            if epoch <= warmup_epochs:
                # 21: Ldom ← 0
                domain_loss = torch.zeros(())
            # 22: else
            else:
                # 23: Forward concatenated features through GRL and the domain classifier to obtain domain logits;
                # 24: Compute domain-adversarial loss Ldom using (15);
                domain_loss = functional.cross_entropy(
                    all_domain_logits, torch.cat([labeled_domains, unlabeled_domains]))
            # 25: end if

            # 26: L = Lsup + λu Lunsup + λdann Ldom
            total_loss = (supervised_loss
                          + unsupervised_weight * unsupervised_loss
                          + domain_adversarial_weight * domain_loss)

            # 27: Update model parameters θ w.r.t. L;
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        # 28: end for
    # 29: end for

    # 30: return θ
    return model