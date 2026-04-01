from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision.models as models
except Exception:
    models = None


class SimpleMobileNet(nn.Module):

    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        if models is not None:
            self.backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate, inplace=True),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(128, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict_with_confidence(self, x: torch.Tensor):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output, dim=1)
            confidence, predictions = probs.max(dim=1)
        return predictions, confidence

    def predict_top_k(self, x: torch.Tensor, k: int = 3):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            probs = F.softmax(output, dim=1)
            top_k_probs, top_k_indices = probs.topk(k, dim=1)
        return top_k_indices, top_k_probs
