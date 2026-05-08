import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def mean_and_std(train):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(train, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.

    for images, _ in loader:
        batch_size = images.size(0)

    # (B, C, H, W) → (B, C, H*W)
        images = images.view(batch_size, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        nb_samples += batch_size

    mean /= nb_samples
    std /= nb_samples

    print("Mean:", mean)
    print("Std:", std)
    return mean, std
meannstd, stdmean = mean_and_std("data_effi/train")
print("Mean:", meannstd)
print("Std:", stdmean)