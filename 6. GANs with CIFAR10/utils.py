import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def dataloader(root, train_batch_size, test_batch_size, num_workers):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
        transforms.ColorJitter(
                brightness=0.1*torch.randn(1),
                contrast=0.1*torch.randn(1),
                saturation=0.1*torch.randn(1),
                hue=0.1*torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])

    trainset = datasets.CIFAR10(root =root, train = True, transform = transform_train, download = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = train_batch_size, shuffle = True, num_workers = num_workers)

    testset = datasets.CIFAR10(root = root, train = False, transform = transform_test, download = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size = test_batch_size, shuffle = False, num_workers = num_workers)

    return trainloader, testloader
