import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os

# ✅ 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = r"C:\Users\disha\OneDrive\Desktop\Sugarcane\Dataset"

# ✅ 2. Transformations (resizing, augmentation, normalization)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ 3. Load dataset
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ✅ 4. Model (Transfer Learning using ResNet18)
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # freeze layers

num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # replace final layer

model = model.to(device)
print(f"Classes found: {train_dataset.classes}")

# ✅ 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ✅ 6. Training loop
epochs = 10
train_loss_history = []

print("\nStarting training...\n")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    train_loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

# ✅ 7. Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/sugarcane_disease_model.pth")
print("\n✅ Model saved to 'models/sugarcane_disease_model.pth'")

# ✅ 8. Plot loss
plt.plot(train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.show()
