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
        err1=self.buildtrainingmodel(no_nodes=25)
        print err1
        err2=self.buildtrainingmodel(no_nodes=50)
        print err2
        err3=self.buildtrainingmodel(no_nodes=75)
        print err3
        err4=self.buildtrainingmodel(no_nodes=100)
        print err4
        err5=self.buildtrainingmodel(no_nodes=125)
        print err5
        err_rate=[err1,err2,err3,err4,err5]
        node_count=[25,50,75,100,125]
        yl = err_rate
        xl = node_count
        plt.title("error/no of nodes in hidden layer")
        plt.scatter(xl, yl)
        plt.savefig("error_rate_vs_noOfNodesTask3.png")
    def buildtrainingmodel(self,no_nodes):
        nn_input_dim = 3072  # input layer dimensionality
        nn_output_dim = 10  # output layer dimensionality
        nn_hdim =no_nodes
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
        a1 = theano.tensor.nnet.relu(n1, alpha=0)
        a2 = a1.dot(W2) + b2
        probabilities = T.nnet.softmax(a2)
        loss_reg = lam * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2)))
        loss=(T.nnet.categorical_crossentropy(probabilities,tar).mean())/num
        mse = T.mean(T.square((probabilities - tar)))
        calerror = theano.function([input_d, tar], mse)
        dW2,db2,dW1,db1 = T.grad(loss, wrt=[W2,b2,W1,b1])
        train = theano.function([input_d,tar],[loss,probabilities],updates=[[W2, W2 - alpha*dW2],
                                                                            [W1, W1 - alpha * dW1],
                                                                            [b2, b2 - alpha * db2],
                                                                            [b1, b1 - alpha * db1]])
        test=theano.function([input_d],probabilities)
        btrain = self.train_data.reshape((self.train_data.shape[0], -1), order='F')
        btest = self.test_data.reshape((self.test_data.shape[0], -1), order='F')
        #train(btrain,self.target_train)
        loss_list=[]
        epoch_count = []
        lo, output = train(btrain, self.target_train)
        err = calerror(btrain, self.target_train)
        return err


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