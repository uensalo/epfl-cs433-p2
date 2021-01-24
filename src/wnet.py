import torch
from torch import nn 

class WNetNode(nn.Module):
    '''
        Implements a node for the WNet(Sha-Net in the paper) network, which 
        performs two convolutions with ReLU activations and batch normalization
    '''
    def __init__(self, in_channels, out_channels, last=False, activation=nn.LeakyReLU(), kernel_size=3, padding=1):
        super(WNetNode,self).__init__()
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.layer1 = nn.Conv2d(in_channels,out_channels, kernel_size=kernel_size,padding=padding)
        self.layer2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.last = last
        
    def forward(self, x):
        o1 = self.layer1(x)
        o1 = self.batchnorm1(o1)
        o1 = self.activation(o1)
        o2 = self.layer2(o1)
        o2 = self.batchnorm2(o2)
        if self.last:
            return o2
        else:
            return self.activation(o2)

class WNet(nn.Module):
        '''
            Implements a WNet, which is a UNet with an extra deconvolutional
            pathway between the skip connections. Consult the paper for a
            visualization of the architecture.
            The size of the input is the same as the size of the output
            given that the initial image dimensions are divisible by 32. 
            The lowest layer converts an NxNx3 image into 'codewords' of size
            N/32xN/32x1024.
        '''
    def __init__(self,deep=False):
        '''
            deep: if set true, the middle deconvolutional layer is connected to
            the loss alongside the last node of the last deconvolutional layer.
        '''
        super(WNet, self).__init__()
        self.x00 = WNetNode(3,  64,  last=False)
        self.x10 = WNetNode(64, 128, last=False)
        self.x20 = WNetNode(128,256, last=False)
        self.x30 = WNetNode(256,512, last=False)
        self.x40 = WNetNode(512,1024,last=False)
        
        self.xm1 = WNetNode(128,  64, last=False)
        self.xm2 = WNetNode(256,  128,last=False)
        self.xm3 = WNetNode(512,  256,last=False)
        self.xm4 = WNetNode(1024, 512,last=False)
        
        self.x31 = WNetNode(1024,512,last=False)
        self.x22 = WNetNode(512, 256,last=False)
        self.x13 = WNetNode(256, 128,last=False)
        self.x04 = WNetNode(128, 64, last=False)
        
        self.up1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.up2 = nn.ConvTranspose2d(512,256, kernel_size=2,stride=2)
        self.up3 = nn.ConvTranspose2d(256,128, kernel_size=2,stride=2)
        self.up4 = nn.ConvTranspose2d(128,64,  kernel_size=2,stride=2)
        
        self.mp1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.mp2 = nn.ConvTranspose2d(512,256, kernel_size=2,stride=2)
        self.mp3 = nn.ConvTranspose2d(256,128, kernel_size=2,stride=2)
        self.mp4 = nn.ConvTranspose2d(128,64,  kernel_size=2,stride=2)
        
        if deep:
            self.deep = True
            self.lst1 = WNetNode(64,1,last=True)
            self.lst2 = WNetNode(64,1,last=True)
            self.lst =  WNetNode(2, 1,last=True)
        
        else:
            self.deep = False
            self.lst = WNetNode(64, 1,last=True)
        
        
    def forward(self,x):
        o1  = self.x00(x)                       #o1 :512x512 x 64
        o1d = nn.MaxPool2d(2)(o1)               #o1d:256x256 x 64
        o2  = self.x10(o1d)                     #o2 :256x256 x 128
        o2d = nn.MaxPool2d(2)(o2)               #o2d:128x128 x 128
        o3  = self.x20(o2d)                     #o3 :128x128 x 256
        o3d = nn.MaxPool2d(2)(o3)               #o3d:64x64   x 256
        o4  = self.x30(o3d)                     #o4 :64x64   x 512
        o4d = nn.MaxPool2d(2)(o4)               #o4d:32x32   x 512
        
        b = self.x40(o4d)                       #b  :32x32   x 1024
        
        #intermediate layers
        mi1 = self.mp1(b)                       #mi1:64x64   x 512
        mo1 = self.xm4(torch.cat([mi1,o4],1))   #mo1:64x64   x 512
        mi2 = self.mp2(mo1)                     #mi2:128x128 x 256
        mo2 = self.xm3(torch.cat([mi2,o3],1))   #mo2:128x128 x 256
        mo2  = nn.Dropout(0.5)(mo2)
        mi3 = self.mp3(mo2)                     #mi3:256x256 x 128
        mo3 = self.xm2(torch.cat([mi3,o2],1))   #mo3:256x256 x 128
        mo3  = nn.Dropout(0.5)(mo3)
        mi4 = self.mp4(mo3)                     #mi4:512x512 x 64
        mo4 = self.xm1(torch.cat([mi4,o1],1))   #mo4:512x512 x 64
        
        o5u = self.up1(b)                       #o5u:64x64   x 512
        o6  = self.x31(torch.cat([o5u,mo1],1))  #o6 :64x64   x 512
        o6  = nn.Dropout(0.5)(o6)
        o6u = self.up2(o6)                      #o6u:128x128 x 256
        o7  = self.x22(torch.cat([o6u,mo2],1))  #o7 :128x128 x 256
        o7  = nn.Dropout(0.5)(o7)
        o7u = self.up3(o7)                      #o7u:256x256 x 128
        o8  = self.x13(torch.cat([o7u,mo3],1))  #o8: 256x256 x 128
        o8  = nn.Dropout(0.5)(o8)
        o8u = self.up4(o8)                      #o8u:512x512 x 64
        o9  = self.x04(torch.cat([o8u,mo4],1))  #o9: 512x512 x 64
        
        if self.deep:
            out1 = self.lst1(o9)
            out2 = self.lst2(mo4)
            out =  self.lst(torch.cat([out1,out2],1))
        else:
            out = self.lst(o9)                      #out:512x512 x 1
            
        out = nn.Sigmoid()(out)                     #logit last layer
        return out