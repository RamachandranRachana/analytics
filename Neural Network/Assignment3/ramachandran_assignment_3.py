#Ramachandran,Rachana
#1001350452
#10/2/2016
#Assignment_03
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

import os
import numpy
path='C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/mnist_images/mnist_images'
noc=10
nod=784
def read_one_image_and_convert_to_vector(file_name):
    img = scipy.misc.imread(file_name).astype(np.float32) # read image and convert to float
    vector = img.reshape(-1,1)/255 # reshape to column vector and return it
    return np.append(vector,1.0)

def get_target_vector(file_name,shape):
	target_vector = np.zeros(shape)
	digit = file_name.split("_")[0]
	digit = int(digit)
	target_vector[digit] = 1
	return np.array(target_vector)
# class ClDataSet:
#     # This class encapsulates the data set
#     # The data set includes input samples and targets
#     def __init__(self, samples=[[0., 0., 1., 1.], [0., 1., 0., 1.]], targets=[[0., 1., 1., 0.]]):
#         # Note: input samples are assumed to be in column order.
#         # This means that each column of the samples matrix is representing
#         # a sample point
#         # The default values for samples and targets represent an exclusive or
#         # Farhad Kamangar 2016_09_05
#         self.samples = np.array(samples)
#
#         if targets != None:
#             self.targets = np.array(targets)
#         else:
#             self.targets = None
#
#
# nn_experiment_default_settings = {
#     # Optional settings
#     "min_initial_weights": -0.1,  # minimum initial weight
#     "max_initial_weights": 0.1,  # maximum initial weight
#     "number_of_inputs": 2,  # number of inputs to the network
#     "learning_rate": 0.1,  # learning rate
#     "momentum": 0.1,  # momentum
#     "batch_size": 0,  # 0 := entire trainingset as a batch
#     "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
#     "data_set": ClDataSet(),
#     'number_of_classes': 3,
#     'number_of_samples_in_each_class': 3
# }

class ClNNGui2d:
    """
    This class presents an experiment to demonstrate
    Perceptron learning in 2d space.
    Farhad Kamangar 2016_09_02
    """

    def __init__(self, master, nn_experiment):
        self.master = master
        self.epoch=0
        self.nn_experiment = nn_experiment
        self.xmin = -5
        self.xmax = 100
        self.ymin = -5
        self.ymax = 100
        self.master.update()

        self.step_size = 10
        self.current_sample_loss = 0
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Multiple Linear Classifiers")
        self.axes = self.figure.add_subplot(111)
        plt.title("Linear Associator")
        plt.scatter(0, 0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Create sliders frame
        self.learning_rate=self.nn_experiment.learning_rate
        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')
        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learning_rate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        self.randomize_weights_button = Tk.Button(self.buttons_frame,
                                                  text="Randomize Weights",
                                                  bg="yellow", fg="red",
                                                  command=lambda: self.randomize_weights_button_callback())
        self.randomize_weights_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights (Learn)",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.hebbian_learning_rule_variable = Tk.StringVar()
        self.hebbian_learning_rule_dropdown = Tk.OptionMenu(self.buttons_frame,
                                                            self.hebbian_learning_rule_variable,
                                                            "Fillearn", "Deltarule", "Unsupervisedhebb",
                                                            command=lambda event: self.hebbian_callback())
        self.hebbian_learning_rule_variable.set("Filllearn")
        self.hebbian_learning_rule_dropdown.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()
        self.refresh_display()

    def initialize(self):
        print "Initialize"

    def hebbian_callback(self):
        self.nn_experiment.hebbian_learning_rule=self.hebbian_learning_rule_variable.get()
        self.refresh_display()
    def display_samples_on_image(self):
        print "display_samples"

    def refresh_display(self):
        print "refresh"
        val=self.nn_experiment.calc_error()
        self.epoch=self.epoch+1
        plt.scatter(self.epoch,val)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)


    def display_neighborhoods(self):
        print "display neighbourhoods"
    def display_output_nodes_net_boundaries(self):
        print "output nodes"
    def learning_rate_slider_callback(self):
        print "learning rate"
        self.learning_rate = self.learning_rate_slider.get()
        self.nn_experiment.learning_rate = self.learning_rate
        self.refresh_display()


    def number_of_classes_slider_callback(self):
        print "no classes"

    def number_of_samples_slider_callback(self):
        print "number of samples slider callback"


    def create_new_samples_bottun_callback(self):
        print "create new samples"


    def adjust_weights_button_callback(self):
        print "adjust weights"
        temp_text = self.adjust_weights_button.config('text')[-1]
        self.adjust_weights_button.config(text='Please Wait')

        self.nn_experiment.adjust_weights()
        self.refresh_display()
        self.adjust_weights_button.config(text=temp_text)
        self.adjust_weights_button.update_idletasks()




    def randomize_weights_button_callback(self):
        print "randomize"

        temp_text = self.randomize_weights_button.config('text')[-1]
        self.randomize_weights_button.config(text='Please Wait')
        self.nn_experiment.randomize_weights(noc,nod)
        self.refresh_display()
        self.randomize_weights_button.config(text=temp_text)
        self.randomize_weights_button.update_idletasks()


    def print_nn_parameters_button_callback(self):
        print "nnparameters"

class ClNNExperiment:

    def __init__(self, path,noc=10,nod=784):
        self.activation_function = 'linear'
        #self.activation_function='linear'

        self.learning_rate = 0.1
        self.gamma = 0.001
        self.hebbian_learning_rule = 'Filtered Learning'
        self.samples = np.array(self.create_samples(path))
        self.targets = np.array(self.get_targets(path,noc))
        self.weights = self.randomize_weights(noc,nod)
        self.activation = self.activation_factor(self.weights,self.samples)

    def create_samples(self, path):
        samp = []
        lfiles=os.listdir(path)
        shuffle(lfiles)
        for filename in lfiles:
            image_vector = read_one_image_and_convert_to_vector(os.path.join(path, filename))
            samp.append(image_vector)

        return samp

    def get_targets(self, path, noc):
        targets = []
        for filename in os.listdir(path):
            target_vector = get_target_vector(filename,noc)
            targets.append(target_vector)

        return targets

    def activation_factor(self, weights, samples):
        activation = []


        for sample in samples:
            activation.append(self.calculate_outputs(weights, sample))
        activation = numpy.array(activation)

        if self.activation_function == 'linear':
            activation = activation
        if self.activation_function == 'sigmoid':
            activation = sigmoid(activation)
        if self.activation_function == 'hardlimit':
            numpy.putmask(activation, activation > 0, 1)
            numpy.putmask(activation, activation <= 0, 0)

        return activation

    def calculate_outputs(self, weights, sample):
        output = numpy.dot( numpy.array(weights), numpy.array(sample) )
        return output

    def randomize_weights(self, noc, nod, min_initial_weights=-1, max_initial_weights=1):
        return numpy.random.uniform(min_initial_weights, max_initial_weights, (noc, nod+1 ))
        self.activation=self.activation_factor()

    def adjust_weights(self):
        if self.hebbian_learning_rule == 'Fillearn':
            print "Filtered Learning"
            sec= self.learning_rate * self.targets.transpose().dot(self.samples)
            first= ((1-self.gamma) * self.weights)
            self.weights = first+sec
        elif self.hebbian_learning_rule == 'Deltarule':
            print "Delta Rule"
            err=self.targets-self.activation
            sec=self.learning_rate*err.transpose().dot(self.samples)
            first=self.weights
            self.weights=first+sec
        elif self.hebbian_learning_rule == 'Unsupervisedhebb':
            print "Unsupervised Hebb"
            sec= self.learning_rate * self.activation.transpose().dot(self.samples)
            first= self.weights
            self.weights = first+sec
        self.activation = self.activation_factor(self.weights, self.samples)

    def calc_error(self):
        errors =0
        n=float(len(self.targets))
        for idx, target in enumerate(self.targets):
            if target.argmax()!= self.activation[idx].argmax():
                errors=errors+1
        return (errors*100) / n

# class ClSingleLayer:
#     """
#     This class presents a single layer of neurons
#     Farhad Kamangar 2016_09_04
#     """
#
#     def __init__(self, settings):
#         self.__dict__.update(single_layer_default_settings)
#         self.__dict__.update(settings)
#         self.randomize_weights()
#
#     def randomize_weights(self, min_initial_weights=None, max_initial_weights=None):
#         if min_initial_weights == None:
#             min_initial_weights = self.min_initial_weights
#         if max_initial_weights == None:
#             max_initial_weights = self.max_initial_weights
#         self.weights = np.random.uniform(min_initial_weights, max_initial_weights,
#                                          (self.number_of_neurons, self.number_of_inputs_to_layer + 1))
#
#     def calculate_output(self, input_values):
#         # Calculate the output of the layer, given the input signals
#         # NOTE: Input is assumed to be a column vector. If the input
#         # is given as a matrix, then each column of the input matrix is assumed to be a sample
#         # Farhad Kamangar Sept. 4, 2016
#         if len(input_values.shape) == 1:
#             net = self.weights.dot(np.append(input_values, 1))
#         else:
#             net = self.weights.dot(np.vstack([input_values, np.ones((1, input_values.shape[1]), float)]))
#         if self.activation_function == 'linear':
#             self.output = net
#         if self.activation_function == 'sigmoid':
#             self.output = sigmoid(net)
#         if self.activation_function == 'hardlimit':
#             np.putmask(net, net > 0, 1)
#             np.putmask(net, net <= 0, 0)
#             self.output = net
#         return self.output


if __name__ == "__main__":
    # nn_experiment_settings = {
    #     "min_initial_weights": -0.1,  # minimum initial weight
    #     "max_initial_weights": 0.1,  # maximum initial weight
    #     "number_of_inputs": 2,  # number of inputs to the network
    #     "learning_rate": 0.1,  # learning rate
    #     "layers_specification": [{"number_of_neurons": 3, "activation_function": "hardlimit"}],  # list of dictionaries
    #     "data_set": ClDataSet(),
    #     'number_of_classes': 2,
    #     'number_of_samples_in_each_class': 3
    # }
    #np.random.seed(1)
    nnexperiment = ClNNExperiment("C:/Users/Rachana/Desktop/UTA/Fall2016/Neural Networks/assignment/mnist_images/mnist_images",10)
    nnexperiment.adjust_weights()
    print nnexperiment.targets.shape
    print len(nnexperiment.weights)
    print len(nnexperiment.weights[0])
    nnexperiment.weights = nnexperiment.randomize_weights(10,784)
    nnexperiment.hebbian_learning_rule = "Deltarule"
    nnexperiment.adjust_weights()
    print nnexperiment.weights.shape
    print nnexperiment.calc_error()
    nnexperiment.weights = nnexperiment.randomize_weights(10,784)
    nnexperiment.hebbian_learning_rule = "Unsupervisedhebb"
    nnexperiment.adjust_weights()
    print nnexperiment.weights.shape
    print nnexperiment.calc_error()
    nnexperiment.weights = nnexperiment.randomize_weights(10, 784)
    nnexperiment.hebbian_learning_rule = "Fillearn"
    nnexperiment.adjust_weights()
    print nnexperiment.weights.shape
    print nnexperiment.calc_error()
    main_frame = Tk.Tk()
    main_frame.title("Perceptron")
    main_frame.geometry('640x480')

    ob_nn_gui_2d = ClNNGui2d(main_frame,nnexperiment)

    main_frame.mainloop()



