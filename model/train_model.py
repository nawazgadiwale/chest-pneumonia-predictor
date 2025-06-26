import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import os

# 🔧 Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 📁 Dataset paths
data_dir = '../../dataset'
train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

# 🔄 Data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 🧠 Model setup
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# ❄️ Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# 🔁 Replace FC layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# 💻 Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 🎯 Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# 🚀 Training loop with validation
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\n🚀 Epoch [{epoch+1}/{epochs}]")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Show batch loss every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  🧪 Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"📉 Epoch [{epoch+1}] Average Loss: {avg_train_loss:.4f}")

    # 🔍 Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")

# 💾 Save model
os.makedirs('saved_models', exist_ok=True)
torch.save(model.state_dict(), 'saved_models/pneumonia_model.pth')
print("✅ Model saved to 'saved_models/pneumonia_model.pth'")
