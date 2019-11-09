import torch
from utils import*
from discriminator import *
from torch.autograd import Variable
import time

root = './cifar10'
batch_size = 128
num_workers = 8
learning_rate = 0.0001
num_epoch = 100


trainloader, testloader = dataloader(root = root, 
                                    train_batch_size= batch_size,
                                    test_batch_size = batch_size,  
                                    num_workers = num_workers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

aD = discriminator()
aD = aD.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(aD.parameters(), lr=0.0001)


train_sum = [ [], [] , []]
test_sum = [ [], [], []]


for epoch in range(num_epoch):
    aD.train()
    epoch_accuracy = 0
    epoch_loss = 0
    batch_counter = 0
    

    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    
    for batch_id, (data, label) in enumerate(trainloader):
        start = time.time()
        data, label = Variable(data).to(device), Variable(label).to(device)
        __, output = aD(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        if(epoch>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()

        pred = torch.max(output,1)[1]
        accuracy = ( float(pred.eq(label).sum()))/ float( batch_size) * 100.0
        epoch_accuracy += accuracy
        epoch_loss += loss.item()
        batch_counter += 1
    end = time.time()

    train_sum[0].append(epoch+1)
    train_sum[1].append(epoch_accuracy/batch_counter)
    train_sum[2].append(epoch_loss/batch_counter)

    print("\nEpoch: {} |Training Accuracy: {} |Trainig Loss: {} | Time: {}".format(train_sum[0][-1], train_sum[1][-1], train_sum[2][-1], (end-start)/60.0))
    
    if (epoch+1)%5 == 0:
        aD.eval()
        epoch_accuracy = 0
        epoch_loss = 0
        batch_counter = 0
        print("\nTESTING\n")

        with torch.no_grad():
            for batch_id, (data, label) in enumerate(testloader):
                start = time.time()
                data, label = Variable(data).to(device), Variable(label).to(device)
                __, output = aD(data)
                loss = criterion(output, label)
                pred = torch.max(output,1)[1]

                accuracy = float(pred.eq(label).sum())/float(batch_size) *100
                epoch_accuracy += accuracy
                epoch_loss += loss.item()
                batch_counter +=1
        end = time.time()

        test_sum[0].append(epoch+1)
        test_sum[1].append(epoch_accuracy/batch_counter)
        test_sum[2].append(epoch_loss/batch_counter)

        print("\nEpoch: {} |Testing Accuracy: {} |Testing Loss: {} |Time: {}".format(test_sum[0][-1], test_sum[1][-1], test_sum[2][-1], (end-start)/60.0))

        torch.cuda.empty_cache()
        torch.save(aD,'cifar10.model')

