import torchvision
from UNET_model import UNET
import torch
import torchvision.transforms as transforms
from affichage import show_result
from PIL import Image
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog

classe = ["normal", "pneumonie"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.4846, 0.4846, 0.4846],
                         [0.2257, 0.2257, 0.2257])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_E = torchvision.models.efficientnet_b0(weights=None).to(device).to(device)
in_features = model_E.classifier[1].in_features
model_E.classifier[1] = nn.Linear(in_features, 2)
model_U = UNET().to(device)

model_E.load_state_dict(torch.load("model.pth", map_location=device))
model_U.load_state_dict(torch.load("best_model.pth", map_location=device))

model_E.eval()
model_U.eval()

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Images", [".png", ".jpg", ".jpeg"])]
)

def image_open(img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    return img

def classe_pred(img_tensor):
    with torch.no_grad():
        output = model_E(img_tensor)
        _, pred = torch.max(output, 1)
    return pred.item()

def unet(img_tensor):
    with torch.no_grad():
        output = model_U(img_tensor)
        mask = torch.sigmoid(output)
    return mask[0].cpu().numpy()

image = image_open(image_path)
pred = classe_pred(image)

print (classe[pred])

mask = None

if classe[pred] == "pneumonie":
    mask = unet(image)

if mask is not None:
    show_result(image, mask)