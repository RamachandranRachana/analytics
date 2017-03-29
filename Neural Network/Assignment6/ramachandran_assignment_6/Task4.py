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
from keras.models import load_model
path_train_data3="C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/assignment6/set3_100"


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
    def __init__(self,train3):
        self.train3=train3.reshape(100,784)

        mymodel=load_model('network.h6')
        fig, axes = plt.subplots(10, 10, figsize=(12, 12))
        fig.suptitle("input", fontsize=16)
        for i in range(100):
            row, column = divmod(i, 10)
            axes[row, column].imshow(self.train3[i,:].reshape(28, 28), cmap=plt.cm.gray)
            axes[row, column].axis('off')
        fig.savefig("InputTask4")
        plt.clf()
        decode=mymodel.predict(self.train3)
        fig2, axes2 = plt.subplots(10, 10, figsize=(12, 12))
        fig2.suptitle("output", fontsize=16)
        for i in range(100):
            row, column = divmod(i, 10)
            axes2[row, column].imshow(decode[i, :].reshape(28, 28), cmap=plt.cm.gray)
            axes2[row, column].axis('off')
        fig2.savefig("OutputTask4")







if __name__ == "__main__":
    train3=create_samples(path_train_data3)
    train3=train3.astype('float32')

    nn=NLayerNeuralNetwork(train3)




