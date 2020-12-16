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

class ShatzNet(nn.Module):
    '''
        Implements a Shatz-Net, which is in essence a WNet (Sha-Net in the
        paper) followed by a UNet.
        The size of the input is the same as the size of the output
        given that the initial image dimensions are divisible by 32. 
        The lowest layer converts an NxNx3 image into 'codewords' of size
        N/32xN/32x1024.
    '''
    
    def __init__(self,deep=False):
        '''
            deep: if set true, the output of the WNet (Sha-Net) is connected
            to the loss function, alongside the output of the UNet.
        '''
        super(ShatzNet, self).__init__()
        
        #WNet
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
        
        self.wlst = WNetNode(64, 1,last=False)
        
        #UNet
        self.y00 = WNetNode(1,  64,  last=False)
        self.y10 = WNetNode(64, 128, last=False)
        self.y20 = WNetNode(128,256, last=False)
        self.y30 = WNetNode(256,512, last=False)
        self.y40 = WNetNode(512,1024,last=False)
        
        self.y31 = WNetNode(1024,512,last=False)
        self.y22 = WNetNode(512, 256,last=False)
        self.y13 = WNetNode(256, 128,last=False)
        self.y04 = WNetNode(128, 64, last=False)
        self.ulst = WNetNode(64,  1,  last=True)
        
        self.yup1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.yup2 = nn.ConvTranspose2d(512,256, kernel_size=2,stride=2)
        self.yup3 = nn.ConvTranspose2d(256,128, kernel_size=2,stride=2)
        self.yup4 = nn.ConvTranspose2d(128,64,  kernel_size=2,stride=2)
        
    def forward(self,x):
        #propagate through wnet
        o1  = self.x00(x)                       #o1 :512x512 x 64
        o1d = nn.MaxPool2d(2)(o1)               #o1d:256x256 x 64
        o2  = self.x10(o1d)                     #o2 :256x256 x 128
        o2d = nn.MaxPool2d(2)(o2)               #o2d:128x128 x 128
        o3  = self.x20(o2d)                     #o3 :128x128 x 256
        o3d = nn.MaxPool2d(2)(o3)               #o3d:64x64   x 256
        o4  = self.x30(o3d)                     #o4 :64x64   x 512
        o4d = nn.MaxPool2d(2)(o4)               #o4d:32x32   x 512
        
        b = self.x40(o4d)                       #b  :32x32   x 1024
        
        #intermediate decoder layers
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
        
        #rightmost decoder layers
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

        out_w = self.wlst(o9)                   #out_w:512x512 x 1
        
        #propagate through unet
        yo1  = self.y00(out_w)                    #yo1 :512x512 x 64
        yo1d = nn.MaxPool2d(2)(yo1)               #yo1d:256x256 x 64
        yo2  = self.y10(yo1d)                     #yo2 :256x256 x 128
        yo2d = nn.MaxPool2d(2)(yo2)               #yo2d:128x128 x 128
        yo3  = self.y20(yo2d)                     #yo3 :128x128 x 256
        yo3d = nn.MaxPool2d(2)(yo3)               #yo3d:64x64   x 256
        yo4  = self.y30(yo3d)                     #yo4 :64x64   x 512
        yo4d = nn.MaxPool2d(2)(yo4)               #yo4d:32x32   x 512
        
        yb = self.x40(yo4d)                        #yb  :32x32   x 1024
        
        yo5u = self.yup1(yb)                       #yo5u:64x64   x 512
        yo6  = self.y31(torch.cat([yo5u,yo4],1))   #yo6 :64x64   x 512
        yo6  = nn.Dropout(0.5)(yo6)
        yo6u = self.yup2(yo6)                      #yo6u:128x128 x 256
        yo7  = self.y22(torch.cat([yo6u,yo3],1))   #yo7 :128x128 x 256
        yo7  = nn.Dropout(0.5)(yo7)
        yo7u = self.yup3(yo7)                      #yo7u:256x256 x 128
        yo8  = self.y13(torch.cat([yo7u,yo2],1))   #yo8: 256x256 x 128
        yo8  = nn.Dropout(0.5)(yo8)
        yo8u = self.yup4(yo8)                      #yo8u:512x512 x 64
        yo9  = self.y04(torch.cat([yo8u,yo1],1))   #yo9: 512x512 x 64
        
        yout = self.ulst(yo9)                     #yout:512x512 x 1
        yout = nn.Sigmoid()(yout)                 #logit last layer
        return yout