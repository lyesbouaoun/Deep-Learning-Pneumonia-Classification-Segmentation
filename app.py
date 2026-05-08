import torchvision
from matplotlib import pyplot as plt
from UNET_model import UNET
import torch
import torchvision.transforms as transforms
from affichage import show_result
from PIL import Image
import torch.nn as nn
import streamlit as st


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

def image_open(img_path):
    img = transform(img_path).unsqueeze(0).to(device)
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

st.set_page_config(
    page_title="U-Net Segmentation et efficientnet classifaire",
    layout="centered",
)

st.title("🧠 U-Net pneumonie Segmentation")

st.write("Upload MRI image")

image_path = st.file_uploader("Upload MRI image",
                         type=["png", "jpg", "jpeg"])
if image_path is not None:

    image = Image.open(image_path).convert('RGB')

    img_tensor = image_open(image)

    pred = classe_pred(img_tensor)

    st.subheader("resultat de la classification")
    st.write(classe[pred])

    if classe[pred] == "pneumonie":
        mask = unet(img_tensor)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))

        img = img_tensor.squeeze()
        img = img.permute(1, 2, 0)

        ax[0].imshow(img)
        ax[0].set_title("image Original")
        ax[0].axis('off')

        ax[1].imshow(mask[0] ,cmap='gray')
        ax[1].set_title("mask")
        ax[1].axis('off')

        ax[2].imshow(img)
        ax[2].imshow(mask[0],cmap='jet',alpha=0.5)
        ax[2].set_title("overlay")
        ax[2].axis('off')

        st.pyplot(fig)



