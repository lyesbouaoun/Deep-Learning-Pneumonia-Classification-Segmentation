import os
from PIL import Image
from torch.utils.data import Dataset

class dataset_u_net_train(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = os.path.join(self.image_dir,self.image_files[index])
        mask = os.path.join(self.mask_dir,self.image_files[index])

        img = Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('L')

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img,mask


class dataset_u_net_val(Dataset):
    def __init__(self,image_dir,mask_dir,transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = os.path.join(self.image_dir,self.image_files[index])
        mask = os.path.join(self.mask_dir,self.image_files[index])

        img = Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('L')

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        return img,mask
