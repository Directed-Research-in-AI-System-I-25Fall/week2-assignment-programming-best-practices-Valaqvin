import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification, ResNetConfig
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
config = ResNetConfig.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def run_inference(model, image):
    model.eval()
    with torch.no_grad():
        inputs = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
        outputs = model(inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class
image, label = mnist_dataset[0]
predicted_class = run_inference(model, image)
print(f"Predicted class: {predicted_class}, Actual label: {label}")