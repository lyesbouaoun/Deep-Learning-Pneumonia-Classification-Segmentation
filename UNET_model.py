import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self,int_c,out_c):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(int_c,out_c,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = double_conv(3,64)
        self.Maxpool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64,128)
        self.Maxpool2 = nn.MaxPool2d(2)
        self.middel = double_conv(128,256)
        self.up1 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.conv1 = double_conv(256,128)
        self.up2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv2 = double_conv(128,64)
        self.out = nn.Conv2d(64,1,1)

    def forward (self,x):
        x1 = self.down1(x)
        x2 = self.Maxpool1(x1)
        x3 = self.down2(x2)
        x4 = self.Maxpool2(x3)
        x5 = self.middel(x4)
        x6 = self.up1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x7 = self.conv1(x6)
        x7 = self.up2(x7)
        x7 = torch.cat([x7,x1],dim=1)
        x8 = self.conv2(x7)
        return self.out(x8)
