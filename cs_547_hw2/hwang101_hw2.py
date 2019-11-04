import numpy as np
import h5py
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('data/MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
x_train = x_train.reshape(-1, 28, 28)
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
x_test = x_test.reshape(-1, 28, 28)
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

class conv:
    def __init__(self, num_filters):
        self.num_filters = num_filters                                     
        self.filters = np.random.randn(self.num_filters,3,3)/9
           
    def define_region (self, image):
        h, w = image.shape
        #Partitioning images that will be convolved with the filters
        for i in range(h-self.num_filters+1):
            for j in range(w-self.num_filters+1):
                filtered_image = image[i:i+3, j:j+3]
                yield filtered_image, i, j

    def forward(self, input):
        self.input = input
        h, w = input.shape
        z = np.zeros((h-self.num_filters+1, w-self.num_filters+1, self.num_filters))

        #Partitioning and convolving images with the filters
        for filtered_image, i, j in self.define_region(input):
            z[i,j]=np.sum(filtered_image *self.filters, axis = (1, 2))
        self.z = z

        return(z)
    def backward(self, dl_dz, lr):
        dl_dk = np.zeros(self.filters.shape)
        #Partial Derivative
        for filtered_img, i, j in self.define_region(self.input):
            for k in range(self.num_filters):
                dl_dk[k] += dl_dz[i,j,k] * filtered_img
        
        #Updating Parameter
        self.filters -= dl_dk*lr        
    

class relu:
    def forward(self, input):
        self.input = input
        a = np.maximum(0, self.input)
        self.a = a
        return a

    def backward(self, dl_da):
        dl_dz = dl_da
        dl_dz[self.a <=0] = 0
        return dl_dz
        

class softmax:
    def __init__(self, input_length, num_class):
        self.weights = np.random.randn(input_length, num_class)/input_length
        self.bias = np.zeros(num_class)

    def forward(self, input):
        self.input_shape = input.shape 
        input = input.flatten()
        self.input = input

        input_length, num_class = self.weights.shape
        
        z = np.dot(input, self.weights) + self.bias
        self.z = z

        #exp = np.exp(z)
        #return exp/np.sum(exp, axis = 0)
        exp = np.exp(z - np.max(z))
        return exp/exp.sum()

    def backward(self, dl_dyh, lr):
        index = dl_dyh != 0 

        exp = np.exp(self.z)
        S = np.sum(exp)

        dyh_dz = -exp[index]*exp/(S**2)
        dyh_dz[index] = exp[index]*(S- exp[index])/(S**2)
        
        dl_dz = dl_dyh[index]*dyh_dz
        dz_dw = self.input
        dz_db = 1
        dz_da = self.weights

        #Partial Derivative
        dl_da =  dz_da @ dl_dz
        dl_db = dl_dz 
        dl_dw = dz_dw[np.newaxis].T@dl_dz[np.newaxis]

        #Updating Parameters
        self.weights -= lr*dl_dw
        self.bias -= lr*dl_db

        return dl_da.reshape(self.input_shape)
    



class loss:
    def __init__(self, prob, y):
        self.y = y
        self.prob = prob
        self.num_class = len(prob)
    def forward(self):
        loss = -np.log(self.prob[y])
        self.loss = loss
        return(loss)

    def backward(self):
        #Partial Derivative
        dl_dyh = np.zeros(self.num_class)
        
        #Updating Parameter
        dl_dyh[self.y] = -1/self.prob[self.y]
        return(dl_dyh)





#init
conv_1 = conv(6)
relu_1 = relu()
smax = softmax(23*23*6, 10)
num_epochs = 3


for epochs in range(num_epochs):
    print("Initializing CNN! Epoch:", epochs +1)
    lr = 0.01
    #Learning rate schedule
    if (epochs > 5):
        lr = 0.001
    if (epochs > 10):
        lr = 0.0001
    if (epochs > 15):
        lr = 0.00001

    accy = 0
    train_accy =0
    loss_1000 = 0
    n = x_train.shape[0]

    for i in range(n):
        n_random = randint(0, n-1)
        x= x_train[n_random]
        y= y_train[n_random]
        
        cov_1_out= conv_1.forward(x) 
        relu_1_out= relu_1.forward(cov_1_out)
        prob = smax.forward(relu_1_out)
        prediction = np.argmax(prob)
        if prediction == y:
            accy += 1
            train_accy +=1
        loss_calc = loss(prob, y)

        if np.isnan(loss_calc.forward()):
            print("probability:", (prob))
            break
        loss_1000 += loss_calc.forward()
        
        dl_dyh = loss_calc.backward()
        dl_da = smax.backward(dl_dyh, lr)
        dl_dz = relu_1.backward(dl_da)
        conv_1.backward(dl_dz, lr) 
        if i%1000 == 999:
            print("Step {}: Past 1000 steps: Average Loss {} | # of correct predictions {}| Accuracy: {}".format(i+1, loss_1000/1000 ,accy, accy/1000))
            loss_1000 = 0 
            accy = 0

    print("Training Epoch #{} finished!\nTraining Accuracy = {} \nTesting model now...".format(epochs+1, train_accy/n))
    ##Testing
    n = x_test.shape[0]
    test_accy = 0

    for i in range(n):
        x = x_test[i]
        y = y_test[i]

        cov_1_out= conv_1.forward(x) 
        relu_1_out= relu_1.forward(cov_1_out)
        prob = smax.forward(relu_1_out)
        prediction = np.argmax(prob)
        if prediction == y:
            test_accy += 1
        loss_calc = loss(prob, y)
        if np.isnan(loss_calc.forward()):
            print("probability:", (prob))
            break
    print("Testing Finished for Epoch #{}!\nModel Accuracy: {}".format(epochs +1, test_accy/n))



        
    



    
