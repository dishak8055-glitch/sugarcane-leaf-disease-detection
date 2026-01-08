import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ================= 1️⃣ Device =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 2️⃣ Model definition =================
num_classes = 11  # must match the number of classes in your trained model
model = models.resnet18(pretrained=False)  # do not load pretrained weights for testing
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ================= 3️⃣ Load saved model =================
model_path = "models/sugarcane_disease_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ================= 4️⃣ Class names =================
# Replace with your actual 11 class names from train_dataset.classes
classes = [
    'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
    'Class6', 'Class7', 'Class8', 'Class9', 'Class10', 'Class11'
]

# ================= 5️⃣ Preprocessing =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================= 6️⃣ Load and preprocess test image =================
image_path = "test_images/sample_leaf.jpg"  # replace with your test image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Test image not found at {image_path}")

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dimension

# ================= 7️⃣ Make prediction =================
with torch.no_grad():
    outputs = model(img_tensor)
    predicted_index = torch.argmax(outputs, dim=1).item()
    predicted_class = classes[predicted_index]

print(f"Predicted class: {predicted_class}")
from torchvision import datasets

data_dir = r"C:\Users\disha\OneDrive\Desktop\Sugarcane\Dataset"
train_dataset = datasets.ImageFolder(root=data_dir)
print("Class-to-Index Mapping:")
print(train_dataset.class_to_idx)
