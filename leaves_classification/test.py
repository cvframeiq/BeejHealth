import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names in the same order as your training set
class_names = ['dead', 'healthy', 'unhealthy']  # adjust if different

# Load the trained model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('resnet50_classification.pth', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and preprocess the image
img_path = 'dataset/val/dead/20231225_124541.jpg_flipped_horizontal.jpg'  # change this to your custom image path
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

print(f'Predicted class: {predicted_class}')
