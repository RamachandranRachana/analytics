import numpy as np
import scipy.misc
import numpy as np
import Tkinter as Tk
import matplotlib
from random import shuffle
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import colorsys
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
import os
import numpy
#Ramachandran,Rachana
#1001350452
#11/14/2016
#Assignment 5

path_train_100="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment5/cifar_data_100_10/train"
path_test_100="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment5/cifar_data_100_10/test"
path_train_1000="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment5/cifar_data_1000_100/train"
path_test_1000="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment5/cifar_data_1000_100/test"

def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
    vector = img.reshape(-1,1)/255 # reshape to column vector and return it
    return vector

def create_samples(path):
    samp = []
    label=[]
    lfiles = os.listdir(path)
    shuffle(lfiles)
    for filename in lfiles:
        label.append(int(filename.split('_')[0]))
        image_vector = read_one_image_and_convert_to_vector(os.path.join(path, filename))
        #normalizing the vector
        samp.append(image_vector)
    return [np.array(samp),label]

class NLayerNeuralNetwork:
    def __init__(self,train,test,trlabels,tslabels):
        self.train_data=train
        self.test_data=test
        self.trlabels=trlabels
        self.tslabels=tslabels
        self.target_train=self.create_target(labels=self.trlabels)
        print self.target_train
        self.initialize()
    def initialize(self):
        self.buildtrainingmodel()
    def buildtrainingmodel(self):
        nn_input_dim = 3072  # input layer dimensionality
        nn_output_dim = 10  # output layer dimensionality
        nn_hdim = 100
        lam = 0.1
        alpha = 0.5
        num = 1000
        input_d= T.dmatrix('input')
        tar = T.dmatrix('target')
        W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
        b1 = theano.shared(np.zeros(nn_hdim), name='b1')
        W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
        b2 = theano.shared(np.zeros(nn_output_dim), name='b2')
        n1 = input_d.dot(W1) + b1
        a1 = theano.tensor.nnet.sigmoid(n1)
        a2 = a1.dot(W2) + b2
        probabilities = T.nnet.softmax(a2)
        loss_reg = lam * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
        loss=(T.nnet.categorical_crossentropy(probabilities,tar).mean())/num
        mse = T.mean(T.square((probabilities - tar)))
        dW2,db2,dW1,db1 = T.grad(loss, wrt=[W2,b2,W1,b1])
        train = theano.function([input_d,tar],[loss,probabilities],updates=[[W2, W2 - alpha*dW2],
                                                                            [W1, W1 - alpha * dW1],
                                                                            [b2, b2 - alpha * db2],
                                                                            [b1, b1 - alpha * db1]])
        test=theano.function([input_d],probabilities)
        calerror = theano.function([input_d, tar], mse)
        btrain = self.train_data.reshape((self.train_data.shape[0], -1), order='F')
        btest = self.test_data.reshape((self.test_data.shape[0], -1), order='F')
        #train(btrain,self.target_train)
        loss_list=[]
        err_list = []
        epoch_count = []
        for i in range(200):
             lo, output = train(btrain,self.target_train)
             print i
             if i % 20 == 0:
                 print "loss: ", lo
                 err = calerror(btrain, self.target_train)
                 epoch_count.append(i)
                 err_list.append(err)
                 loss_list.append(round(lo,5))
        print "Testing the data"
        output_test = test(btest)
        print output_test.shape
        print len(self.tslabels)
        tv=self.tslabels
        av=[]
        for i in range(100):
            o=output_test[i].tolist()
            max_val=max(o)
            a=o.index(max_val)
            av.append(a)
        print confusion_matrix(tv,av)
        plt.matshow(confusion_matrix(tv,av))
        plt.savefig("confusion_mat_task2.png")
        plt.clf()
        yl = err_list
        xl = epoch_count
        plt.title("error/epoches")
        plt.scatter(xl, yl)
        plt.savefig("error_rate_task2.png")
        plt.clf()
        yl=loss_list
        xl=epoch_count
        plt.title("loss/epoches")
        plt.scatter(xl, yl)
        plt.savefig("loss_rate_task2.png")



    def create_target(self,labels):
        target=[]
        for i in range(len(labels)):
            t=[0.0]*10
            y=labels[i]
            t[y]=1.0
            target.append(t)
            t = [0.0] * 10
        return np.array(target)

if __name__ == "__main__":
    train_labels=create_samples(path_train_100)
    train=train_labels[0]
    trlabels=train_labels[1]
    test_labels=create_samples(path_test_100)
    test = test_labels[0]
    tslabels = test_labels[1]
    nn=NLayerNeuralNetwork(train,test,trlabels,tslabels)