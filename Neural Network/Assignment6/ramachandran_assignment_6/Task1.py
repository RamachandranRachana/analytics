from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
import matplotlib.pyplot as plt
import colorsys
from sklearn.metrics import confusion_matrix
import theano
import theano.tensor as T
import os
import numpy
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Input, Dense
from keras.layers.core import Dense, Activation
path_train_data1="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment6/set1_20k/train"
path_train_data2="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment6/set2_2k"

# function to read an image
def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
    vector = img.reshape(-1,1)/255 # reshape to column vector and return it
    return vector
def create_samples(path):
    samp = []
    lfiles = os.listdir(path)
    shuffle(lfiles)
    for filename in lfiles:
        image_vector = read_one_image_and_convert_to_vector(os.path.join(path, filename))
        samp.append(image_vector)
    return np.array(samp)

class NLayerNeuralNetwork:
    def __init__(self,train1,train2):
        self.nbclass=784
        self.n_hidden=100
        self.train1=train1.reshape(20000,784)
        self.train2=train2.reshape(2000,784)
        self.input_shape=784
        model = Sequential()
        model.add(Dense(self.n_hidden, input_shape=(784,), activation='relu'))
        model.add(Dense(self.nbclass, activation='linear'))
        self.batch_size = 128
        self.nb_epoch = 50
        self.optimizer = 'RMSprop'
        self.loss = 'mean_squared_error'
        self.metrics = ['accuracy']
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        history=model.fit(self.train1,self.train1,batch_size=self.batch_size, nb_epoch=self.nb_epoch,verbose=2, validation_data=(self.train2,self.train2))
        train_handle, = plt.plot(history.history['loss'], label='train1')
        validation_handle, = plt.plot(history.history['val_loss'], label='train2')
        plt.legend(handles=[train_handle, validation_handle])

        plt.xlabel("epochs")
        plt.ylabel("mean_square_error")
        plt.suptitle("Loss")
        plt.savefig("Loss_Task1")






if __name__ == "__main__":
    train1=create_samples(path_train_data1)
    train1=train1.astype('float32')
    train2=create_samples(path_train_data2)
    train2=train2.astype('float32')
    nn=NLayerNeuralNetwork(train1,train2)


