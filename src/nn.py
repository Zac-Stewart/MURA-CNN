import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=1):#
        super().__init__()
        self.feature_extractor = backbone
        self.classifier = self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),  # Hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),      # Regularization
            nn.Linear(256, 64),   # Another hidden layer
            nn.ReLU(),
            nn.Linear(64, num_classes)      # Final output layer
        )

    def forward(self, images):
        batch_size, num_images, C, H, W = images.shape
        images = images.view(batch_size * num_images, C, H, W)

        features = self.feature_extractor(images)
        features = features.view(batch_size, num_images, -1)

        pooled_features = features.max(dim=1)[0]

        logits = self.classifier(pooled_features)

        return logits