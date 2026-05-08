import torch
import torchvision.transforms as transforms
from efficientnet_model import efficientnet
import pandas as pd
from dataset_efficientnet import dataset
import torch.optim as optim
from affichage import affich_graph,confus_matrix
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

classe = ["normal" , " pneumonie "]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])


train, val = dataset(
    image_train='./data_effi/train',
    image_val='./data_effi/val'
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet().to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=3,
    factor=0.5
)

train_loss=[]
val_loss=[]
train_accs = []
val_accs = []
best_val = float("inf")
for epoch in range(30):

    model.train()

    correct = 0
    total = 0
    error = 0

    for image, label in train:

        image, label = image.to(device), label.to(device)

        output = model(image)
        Loss = loss(output, label)

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        error += Loss.item()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (label == predicted).sum().item()
    train_loss_epoch = error / len(train)
    accuracy_train = correct / total

    model.eval()

    error_val = 0
    total_val = 0
    correct_val = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for image, label in val:

            image, label = image.to(device), label.to(device)

            output = model(image)
            Loss = loss(output, label)
            error_val += Loss.item()
            _, predicted = torch.max(output.data, 1)
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            total_val += label.size(0)
            correct_val += (label == predicted).sum().item()
    val_loss_epoch = error_val / len(val)
    accuracy_val = correct_val / total_val
    scheduler.step(val_loss_epoch)
    if val_loss_epoch < best_val:
        best_val = val_loss_epoch

        torch.save(model.state_dict(), "best_model.pth")

    train_loss.append(train_loss_epoch)
    val_loss.append(val_loss_epoch)
    train_accs.append(accuracy_train)
    val_accs.append(accuracy_val)

    print(f"train loss: {train_loss_epoch:.4f}")
    print(f"val loss: {val_loss_epoch:.4f}")
    print(f"train acc: {accuracy_train:.2f}")
    print(f"val acc: {accuracy_val:.2f}")

df = pd.DataFrame ({
    "epochs" : list(range(1 , len(train_loss)+1)),
    "train_loss": train_loss,
    "val_loss" : val_loss,
    "train_acc": train_accs,
    "val_acc": val_accs
    })
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
error_val = 0
total_val = 0
correct_val = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for image, label in val:
        image, label = image.to(device), label.to(device)

        output = model(image)
        Loss = loss(output, label)
        error_val += Loss.item()
        _, predicted = torch.max(output.data, 1)
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        total_val += label.size(0)
        correct_val += (label == predicted).sum().item()
precision = precision_score(all_labels, all_predictions, zero_division=0)
recall = recall_score(all_labels, all_predictions, zero_division=0)
f1 = f1_score(all_labels, all_predictions, zero_division=0)
print(classification_report(
    all_labels,
    all_predictions,
    target_names=classe,
    zero_division=0
))
cm = confusion_matrix(all_labels, all_predictions)
confus_matrix (cm , classe)
df.to_csv("training_log.csv", index=False)

affich_graph(train_loss,val_loss,train_accs,val_accs)





