import torch
from torch import nn 

class UNet(nn.Module):
    '''
        Implements a UNet with 4 downsampling layers, and 4 deconvolutional
        layers. The size of the input is the same as the size of the output
        given that the initial image dimensions are divisible by 32. The lowest
        layer converts an NxNx3 image into 'codewords' of size N/32xN/32x1024.
    '''
    def __init__(self):
        super(UNet, self).__init__()
        self.down_layer1 = nn.Sequential(
            nn.Conv2d(3, 64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        
        self.down_layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        
        self.down_layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        
        self.down_layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
        )
        
        self.bottom_layer = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(512,1024,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(1024,1024,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        )
        
        self.up_layer_1 = nn.Sequential(
            nn.Conv2d(1024,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        )
        
        self.up_layer_2 = nn.Sequential(
            nn.Conv2d(512,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        )
        
        self.up_layer_3 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        )
        
        self.up_layer_4 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64,1,kernel_size=1),
            #nn.Sigmoid()
        )
        
    def forward(self,x1):
        o1 = self.down_layer1(x1)
        o2 = self.down_layer2(o1)
        o3 = self.down_layer3(o2)
        o4 = self.down_layer4(o3)
        
        b1  = self.bottom_layer(o4)

        u1 = self.up_layer_1(torch.cat([b1,o4], 1))
        u2 = self.up_layer_2(torch.cat([u1,o3], 1))
        u3 = self.up_layer_3(torch.cat([u2,o2], 1))
        u4 = self.up_layer_4(torch.cat([u3,o1], 1))
        return u4