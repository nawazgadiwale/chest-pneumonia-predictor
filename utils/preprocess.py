from PIL import Image
from torchvision import transforms
def preprocess(img_path):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)
