import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(in_features = 100, out_features = 196*4*4)
        self.bnf = nn.BatchNorm2d(196*4*4)

        self.conv1 = nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn1 = nn.BatchNorm2d(196)

        self.conv2 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(196)

        self.conv3 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(196)

        self.conv4 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn4 = nn.BatchNorm2d(196)

        self.conv5 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn5 = nn.BatchNorm2d(196)

        self.conv6 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn6 = nn.BatchNorm2d(196)

        self.conv7 = nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn7= nn.BatchNorm2d(196)

        self.conv8 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.bnf(x)
        x = x.view(-1, 196, 4, 4)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = self.bn6(x)
        x = F.relu(self.conv7(x))
        x = self.bn7(x)
        x = F.relu(self.conv8(x))

        return x