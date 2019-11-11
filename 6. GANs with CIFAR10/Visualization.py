import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable


from utils import*
from discriminator import *
from generator import *

import time
import numpy as np


'''Loading Data and Model'''

batch_size = 128

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
])

testset = datasets.CIFAR10(root = './cifar10', train = False, download = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 8)
testloader = enumerate(testloader)

model = torch.load('cifar10.model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

batch_indx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch, requires_grad = True).to(device)
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).to(device)
Y_batch = Variable(Y_batch).to(device)



##Saving the first 100 real images with alternative labels in visualization
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_img.png', bbox_inches = 'tight')
plt.close(fig)


#Outputting accuracy from fc10 layer
_, output = model(X_batch)
prediction = output.data.max(1)[1]
accuracy = ( float( prediction.eq(Y_batch.data).sum() )  /float(batch_size)) * 100.0
print(accuracy)

#Calculating gradients with alternate label
criterion = nn.CrossEntropyLoss(reduce = False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs = loss, inputs = X_batch,
                            grad_outputs = torch.ones(loss.size()).to(device),
                            create_graph = True, retain_graph = False, only_inputs = True)[0]

## save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image_cifar.png', bbox_inches = 'tight')
plt.close(fig)

#Calculating loss based on the alternative classes instead of the real classes"

##jitter input image
gradients[gradients > 0.0] = 1.0
gradients[gradients < 0.0] = -1.0

gain =8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified > 1.0] = 1.0
X_batch_modified[X_batch_modified < -1.0] = -1.0

## Evaluating new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1]
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size) ) *100.0
print(accuracy)


## Save Jitttered Images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images_cifar.png', bbox_inches = 'tight')
plt.close(fig)



'''Synthetic Images Maximizing Classification Output'''
#Calculating the mean image and making a copy for each clss.
X = X_batch.mean(dim = 0)
X = X.repeat(10, 1, 1, 1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).to(device)


#Synthetic images maximizing class output for discriminator
lr = 0.1
weight_decay = 0.001

for i in range(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs = loss, inputs = X,
                                    grad_outputs = torch.ones(loss.size()).to(device),
                                    create_graph = True, retain_graph = False, only_inputs = True)[0]

    prediction = output.data.max(1)[1]
    accuracy = ( float(prediction.eq(Y.data).sum() )/ float(10.0)) *100.0
    print("Iteration: {} | Accuracy: {} | Loss: {} ".format(i,accuracy, -loss))

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] =-1.0

samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class.png_cifar', bbox_inches='tight')
plt.close(fig)


'''Synthetic Features Maximizing Features at Various Layers '''
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).to(device)

lr = 0.1
weight_decay = 0.001
for j in range(1,9):
    print("feature:", j)
    
    for i in range(200):
        output = model(X, extract_features = j)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).to(device),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print("Iteration: {} | Accuracy: {} | Loss: {} ".format(i,accuracy, -loss))

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
                    
    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features_{}_cifar.png'.format(j), bbox_inches='tight')
    plt.close(fig)

