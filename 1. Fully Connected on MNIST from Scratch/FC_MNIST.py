import numpy as np
import h5py
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

'''
import matplotlib
import matplotlib.pyplot as plt

for i in range(5):
    plt.imshow(x_train[i].reshape(28,28), cmap = matplotlib.cm.binary)
    plt.axis("off")
    plt.show()
    print(i)
'''


#Initialize
n_h = 64
n_x = x_train.shape[1]
n_class = len(np.unique(y_train))

w = {}
w['W1'] = np.random.randn(n_h, n_x)/np.sqrt(n_x)
w['W2'] = np.random.randn(n_class, n_h)/np.sqrt(n_h)
dw = copy.deepcopy(w)

def sigmoid (z):
    s = 1/( 1 + np.exp(-z) )
    return(s)

def softmax (s):
    smax = np.exp(s)/np.sum(np.exp(s), axis = 0)
    return(smax)

def forward (x, w):
    z1 = np.matmul(w['W1'], x).reshape(-1,1) 
    a1 = sigmoid(z1)
    z2 = np.matmul (w['W2'], a1 )
    p = softmax(z2)
    return (p)

def backward (x, y, p, w):
    dz2 = p
    dz2[y] = dz2[y] - 1
    s = sigmoid(np.matmul(w['W1'],x).reshape(-1,1))
    dw2 = np.matmul(dz2, s.T)
    da1 = np.matmul(w['W2'].T, dz2)
    dz1 = da1 * s * (1-s)
    dw1 = np.matmul(dz1, x.reshape(-1,1).T)

    return dw1, dw2

LR = 0.01
num_epochs = 20

for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        #Stochastic Gradient Descent
        n_random = randint(0, len(x_train)-1)
        x = x_train[n_random]
        y = y_train[n_random]
        p = forward(x, w)
        prediction = np.argmax(p)
        if prediction == y:
            total_correct += 1
        dw['W1'], dw['W2'] = backward(x, y, p, w)
        w['W1'] = w['W1'] - LR*dw['W1']
        w['W2'] = w['W2'] - LR*dw['W2']
    
    print ("Epoch:", epochs, "\nTraining Accuracy:",total_correct / len(y_train))
    #Testing Accuracy
    total_correct = 0
    for n in range( len(x_test)):
        x = x_train[n]
        y = y_train[n]
        p = forward(x,w)
        prediction = np.argmax(p)
        if prediction == y:
            total_correct += 1
    print ("Testing Accuracy:",total_correct / len(y_test))


