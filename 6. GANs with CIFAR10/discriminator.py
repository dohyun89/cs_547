import torch
import torch.nn as nn
import torch.nn.functional as F

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.lnorm1 = nn.LayerNorm([32,32])
        self.leaky1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =2)
        self.lnorm2 = nn.LayerNorm([16,16])
        self.leaky2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =1)
        self.lnorm3 = nn.LayerNorm([16,16])
        self.leaky3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =2)
        self.lnorm4 = nn.LayerNorm([8,8])
        self.leaky4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =1)
        self.lnorm5 = nn.LayerNorm([8,8])
        self.leaky5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =1)
        self.lnorm6 = nn.LayerNorm([8,8])
        self.leaky6 = nn.LeakyReLU()
        
        self.conv7 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =1)
        self.lnorm7 = nn.LayerNorm([8,8])
        self.leaky7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride =2)
        self.lnorm8 = nn.LayerNorm([4,4])
        self.leaky8 = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(4, stride = 4)

        self.fc1 = nn.Linear(in_features = 196, out_features = 1)
        self.fc10 = nn.Linear(in_features = 196, out_features = 10)

    def forward(self,x, extract_features = 0):
        x = self.conv1(x)
        if(extract_features==1):
            h = F.max_pool2d(x,32,32)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm1(x)
        x = self.leaky1(x)

        x = self.conv2(x)
        if(extract_features==2):
            h = F.max_pool2d(x,16,16)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm2(x)
        x = self.leaky2(x)

        x = self.conv3(x)
        if(extract_features==3):
            h = F.max_pool2d(x,16,16)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm3(x)
        x = self.leaky3(x)

        x = self.conv4(x)
        if(extract_features==4):
            h = F.max_pool2d(x,8,8)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm4(x)
        x = self.leaky4(x)

        x = self.conv5(x)
        if(extract_features==5):
            h = F.max_pool2d(x,8,8)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm5(x)
        x = self.leaky5(x)

        x = self.conv6(x)
        if(extract_features==6):
            h = F.max_pool2d(x,8,8)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm6(x)
        x = self.leaky6(x)

        x = self.conv7(x)
        if(extract_features==7):
            h = F.max_pool2d(x,8,8)
            h = h.view(-1, 196)
            return h        
        x = self.lnorm7(x)
        x = self.leaky7(x)

        x = self.conv8(x)
        if(extract_features==8):
            h = F.max_pool2d(x,4,4)
            h = h.view(-1, 196)
            return h

        x = self.lnorm8(x)
        x = self.leaky8(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        out_1 = self.fc1(x)
        out_10 = self.fc10(x)

        return out_1, out_10
