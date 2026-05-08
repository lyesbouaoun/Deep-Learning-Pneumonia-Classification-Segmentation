import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

from UNET_model import UNET
from dataset_unet import dataset_u_net_train, dataset_u_net_val
from affichage import affich_graph


# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# DATASET

train_dataset = dataset_u_net_train(
    image_dir = './dataset/train/image',
    mask_dir = './dataset/train/mask',
    transform = transform
)

val_dataset = dataset_u_net_val(
    image_dir = './dataset/val/image',
    mask_dir = './dataset/val/mask',
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# MODEL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET().to(device)

bce = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

# LOSS DICE

class DiceLoss(torch.nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()

        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

        return 1 - dice

dice_loss = DiceLoss()

# -------------------
# METRICS
# -------------------
def dice_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def iou_score(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection + 1e-6

    return intersection / union


train_loss_list = []
val_loss_list = []
dice_list = []
iou_list = []

best_val = float("inf")

for epoch in range(20):

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0

    for image, mask in train_loader:
        image = image.to(device)
        mask = mask.to(device).float()

        output = model(image)

        loss = bce(output, mask) + dice_loss(output, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0
    dice_total = 0
    iou_total = 0

    with torch.no_grad():
        for image, mask in val_loader:
            image = image.to(device)
            mask = mask.to(device).float()

            output = model(image)

            loss = bce(output, mask) + dice_loss(output, mask)
            val_loss += loss.item()

            pred = (torch.sigmoid(output) > 0.5).float()

            dice_total += dice_score(pred, mask).item()
            iou_total += iou_score(pred, mask).item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    if val_loss < best_val:
        best_val = val_loss

        torch.save(model.state_dict(), "best_model.pth")
    dice_avg = dice_total / len(val_loader)
    iou_avg = iou_total / len(val_loader)

    # ---------------- SAVE METRICS ----------------
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    dice_list.append(dice_avg)
    iou_list.append(iou_avg)

    # ---------------- PRINT ----------------
    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Dice Score : {dice_avg:.4f}")
    print(f"IoU Score  : {iou_avg:.4f}")

# DATAFRAME
df = pd.DataFrame({
    "epoch": list(range(1, len(train_loss_list)+1)),
    "train_loss": train_loss_list,
    "val_loss": val_loss_list,
    "dice": dice_list,
    "iou": iou_list
})

# PLOT

affich_graph(train_loss_list, val_loss_list, dice_list, iou_list)