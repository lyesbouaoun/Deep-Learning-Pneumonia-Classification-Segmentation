import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def affich_graph(train_losses, val_losses, train_accs, val_accs):
# Loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

# Accuracy
    plt.figure()
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.show()

def confus_matrix(cm, classe):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classe,
                yticklabels=classe)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def show_result(image,mask):
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 0.2257) + 0.4846
    image_np = np.clip(image_np, 0, 1)
    plt.figure(figsize=(6, 5))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(image_np)
    plt.subplot(1,3,2)
    plt.title("predict Mask")
    plt.imshow(mask[0])
    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(image_np)
    plt.imshow(mask[0],cmap="jet",alpha=0.5)
    plt.show()