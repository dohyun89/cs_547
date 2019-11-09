import torch
import torch.nn as nn
import torch.nn.functional as F

class generator(nn.Module):
    def __init__(self):
    
        super(Generator, self).__init__()

        #Input: N x 100
        self.fc1 = nn.Linear(100, 196 * 4 * 4)

        # Input: N x 196 x 4 x 4
        self.conv1 = nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn1 = nn.BatchNorm2d(196)

        # Input: N x 196 x 8 x 8 
        self.conv2 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn2 = nn.BatchNorm2d(196)

        # Input: N x 196 x 8 x 8 
        self.conv3 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(196)

        # Input: N x 196 x 8 x 8 
        self.conv4 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn4 = nn.BatchNorm2d(196)

        # Input: N x 196 x 8 x 8
        self.conv5 = nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn5 = nn.BatchNorm2d(196)

        # Input: N x 196 x 16 x 16
        self.conv6 = nn.Conv2d(in_channels = 196, out_channels = 196, kernel_size = 3, padding = 1, stride = 1)
        self.bn6 = nn.BatchNorm2d(196)

        # Input: N x 196 x 16 x 16
        self.conv7 = nn.ConvTranspose2d(in_channels = 196, out_channels = 196, kernel_size = 4, padding = 1, stride = 2)
        self.bn7 = nn.BatchNorm2d(196)

        # Input: N x 196 x 32 x 32
        self.conv8 = nn.Conv2d(in_channels = 196, out_channels = 3, kernel_size = 3, padding = 1, stride = 1)

        # Output: N x 3 x 32 x 32

    def forward(self, x):

        x = self.fc1(x)
        x = x.reshape(-1, 196, 4, 4)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.conv8(x)

        return F.tanh(x)
