from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from mean_std_data import mean_and_std

def dataset (image_train,image_val,):
    mean,std = mean_and_std(image_train)
    transform_train = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(10,(0.05,0.05),(0.9,1.1)),
        transforms.ToTensor(),
        transforms.Normalize( mean=mean, std=std ),
        ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


    data_train=ImageFolder(root=image_train,transform=transform_train)
    train_loader = DataLoader(dataset=data_train,batch_size=16,shuffle=True)

    data_val = ImageFolder(root=image_val, transform=transform_val)
    val_loader = DataLoader(dataset=data_val, batch_size=16, shuffle=True)

    return train_loader,val_loader




