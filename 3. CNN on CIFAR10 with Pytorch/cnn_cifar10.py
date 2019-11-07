import torch
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, padding = 2),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p = 0.1),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, padding =2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p = 0.1),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p = 0.1),

            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Dropout2d(p = 0.1)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(64*4*4,500),
            nn.ReLU(inplace = True),
            nn.Linear(500, 500),
            nn.ReLU(inplace = True),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0),-1)

        x = self.fc_layer(x)

        return(x)


device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
device
net = Net().to(device)
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, weight_decay=1e-5)

import numpy as np
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            
    
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10,10], int)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1 

    model_accuracy = total_correct / total_images * 100
    print('Model accuracy for epoch # {0}: {1:.2f}%'.format(epoch + 1, model_accuracy))
print('Finished Training.')
torch.save(net.state_dict(), 'cifar10_hwang.ckpt')