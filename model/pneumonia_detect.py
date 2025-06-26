import torch
from torchvision import transforms, models
from PIL import Image

def load_model(model_path):
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes: NORMAL, PNEUMONIA
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_image(image_path):
    model_path = "model/saved_models/pneumonia_model.pth"
    model = load_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.colorjitter(brightness = 0.4),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        class_idx = torch.argmax(probs).item()

    classes = ['NORMAL', 'PNEUMONIA']
    return classes[class_idx], float(probs[class_idx])
