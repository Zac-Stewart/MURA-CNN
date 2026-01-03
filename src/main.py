import time

import torch
from sympy.stats.rv import probability
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context


from MURA_Dataset import MURA_Dataset
from nn import Classifier

learning_rate = 0.001
batch_size = 64
epochs = 10
progress_interval = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.176], std=[0.139]),
    transforms.Lambda(lambda x: torch.clamp(x, min=-2.25, max=2.25))
])
train_dataset = MURA_Dataset(csv_images='MURA-v1.1/train_image_paths.csv', csv_labels='MURA-v1.1/train_labeled_studies.csv',
                                    root_dir='', transform=transform, device=device)
#test_dataset = train_dataset
test_dataset = MURA_Dataset(csv_images='MURA-v1.1/valid_image_paths.csv', csv_labels='MURA-v1.1/valid_labeled_studies.csv',
                                    root_dir='', transform=transform, device=device)
'''
figure = plt.figure(figsize=(8, 8))
cols, rows = 10, 10
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    print(img.shape)
    plt.imshow(img[0].squeeze(), cmap="gray")
plt.show()
'''


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)





from torchvision.models import resnet18, ResNet18_Weights

backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
backbone.fc == nn.Identity()
backbone.conv1 = nn.Conv2d(
    in_channels=1,  # Change from 3 to 1
    out_channels=backbone.conv1.out_channels,
    kernel_size=backbone.conv1.kernel_size,
    stride=backbone.conv1.stride,
    padding=backbone.conv1.padding,
    bias=backbone.conv1.bias is not None
)
feature_dim = 1000

model = torch.jit.script(Classifier(backbone, feature_dim))
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss()

def train_model(model, train_loader, test_loader, num_epochs=10, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()  # Set model to training mode
        total_loss = 0
        total_batches = len(train_loader)
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            logits = model(images).squeeze(1)  # Shape: (batch_size,)
            loss = loss_fn(logits, labels)  # BCEWithLogitsLoss expects raw logits
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % progress_interval == 0 or batch_idx == total_batches - 1:
                percent_complete = (batch_idx + 1) / total_batches * 100
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx + 1}/{total_batches}] "
                      f"({percent_complete:.1f}%) - Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / num_batches
        print(f"Epoch [{epoch}/{num_epochs}] completed. Average Training Loss: {avg_train_loss:.4f}")

        # Evaluate on test set
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy after Epoch {epoch}: {test_accuracy:.2f}%\n")

        torch.save(model.state_dict(), 'model_weights.pth')

def evaluate_model(model, test_loader, device="cuda"):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradients needed during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            predictions = (logits > 0).float()  # Convert logits to binary predictions
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


print(evaluate_model(model, test_dataloader, device))
#train_model(model, train_dataloader, test_dataloader, epochs, learning_rate, device)