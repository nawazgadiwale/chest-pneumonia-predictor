from sklearn.metrics import classification_report
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pneumonia_detect import load_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder('../../dataset/test', transform=transform)
test_loader = DataLoader(test_data, batch_size = 32)

model = load_model('saved_models/pneumonia_model.pth')
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim = 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))