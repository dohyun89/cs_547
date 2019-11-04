import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from RNN_model import RNN_model

#glove_embeddings = np.load('../../preprocessed_data/glove_embeddings.npy')
vocab_size = 8000
num_hidden_units = 300
seq_length = 300

x_train = []
with io.open('../../preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)

x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = RNN_model(vocab_size, num_hidden_units)
model.cuda()


##Test and Train
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)


batch_size = 200
no_of_epochs = 20
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []
test_loss = []

for epoch in range(no_of_epochs):
    
    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

    ## within the training loop
        x_input2 = [x_train[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = seq_length
        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target,train=True)
        loss.backward()

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size
    

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print("Train Epoch: ",epoch+1, "\nAccu: %.2f" % (epoch_acc*100.0), "Loss: %.4f" % epoch_loss, "Time: %.4f" % float(time.time()-time1))


        # do testing loop
        # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        sequence_length = seq_length
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j, :] = x[start_index:(start_index+sequence_length)]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data,target, train = False)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    scheduler.step()

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_loss.append(epoch_loss)
    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("Test Epoch:", epoch+1, "\nAccu: %.2f" % (epoch_acc*100.0), "Loss: %.4f" % epoch_loss)

torch.save(model,'RNN.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data.npy',data)

#print("Training Accuracy: {}\nTraining Loss: {}\nTesting Accuracy: {}\nTesting Loss:{}".format(train_accu,train_loss,test_accu,test_loss))
