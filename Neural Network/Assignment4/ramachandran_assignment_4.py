#Ramachandran,Rachana
#1001350452
#10/16/2016
#Assignment 4
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
from decimal import Decimal
def read_csv_as_matrix(file_name):
    # Each row of data in the file becomes a row in the matrix
    # So the resulting matrix has dimension [num_samples x sample_dimension]
    data = np.loadtxt(file_name, skiprows=1, delimiter=',', dtype=np.float32)
    return data
class ClNNGui2d:
    def __init__(self, master,stockdata):
        self.master = master
        self.step_size = 10
        self.stockdata=stockdata

        self.xmin =0
        self.xmax =10
        self.ymin =-100
        self.ymax = 100
        self.batch_count=0
        self.number_of_delayedelements=3
        self.learningRate=0.8
        self.sampleSizePerc=20
        self.batchSize=20
        self.iterations=2

        prize = self.stockdata[:, 0]/214.0
        volume = self.stockdata[:, 1]/16337400
        p = [round(i,2) for i in prize.tolist()]
        v = [round(i,2) for i in volume.tolist()]
        self.totalsize=self.stockdata.shape[0]
        self.current_sample_size = (self.totalsize * self.sampleSizePerc) / 100
        self.current_batch_size = (self.current_sample_size * self.batchSize) / 100
        self.sample_p = p[:self.current_sample_size]
        self.sample_v = v[:self.current_sample_size]
        self.master.update()
        self.master.rowconfigure(0, weight=2, uniform="group1")
        self.master.rowconfigure(1, weight=1, uniform="group1")
        self.master.columnconfigure(0, weight=2, uniform="group1")
        self.master.columnconfigure(1, weight=1, uniform="group1")

        self.canvas = Tk.Canvas(self.master)
        self.display_frame = Tk.Frame(self.master)
        self.display_frame.grid(row=0, column=0, columnspan=2, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.figure = plt.figure("Mean Squared Error (MSE) and Maximum Absolute error")
        self.axes = self.figure.add_subplot(111)
        self.figure = plt.figure("Mean Squared Error (MSE) and Maximum Absolute error")
        self.axes = self.figure.add_subplot(111)
        plt.title("Mean Squared Error (MSE) and Maximum Absolute error")
        plt.scatter(0, 0)
        # plt.scatter(30,0)
        plt.xlim(self.xmin, self.xmax)
        plt.ylim(self.ymin, self.ymax)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # Create buttons frame
        self.buttons_frame = Tk.Frame(self.master)
        self.buttons_frame.grid(row=1, column=1)
        self.buttons_frame.rowconfigure(0, weight=1)
        self.buttons_frame.columnconfigure(0, weight=1, uniform='b1')
        # Create sliders frame

        self.sliders_frame = Tk.Frame(self.master)
        self.sliders_frame.grid(row=1, column=0)
        self.sliders_frame.rowconfigure(0, weight=10)
        self.sliders_frame.rowconfigure(1, weight=2)
        self.sliders_frame.columnconfigure(0, weight=1, uniform='s1')
        self.sliders_frame.columnconfigure(1, weight=1, uniform='s1')

        # Number of delayed elements slider
        self.number_of_delayedelements_slider_label = Tk.Label(self.sliders_frame, text="Number of Delayed elements")
        self.number_of_delayedelements_slider_label.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.number_of_delayedelements_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                                         from_=2, to_=20, bg="#DDDDDD",
                                                         activebackground="#FF0000",
                                                         highlightcolor="#00FFFF", width=10)
        self.number_of_delayedelements_slider.set(self.number_of_delayedelements)
        self.number_of_delayedelements_slider.bind("<ButtonRelease-1>",
                                                   lambda event: self.number_of_delayedelements_slider_callback())
        self.number_of_delayedelements_slider.grid(row=1, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)

        # Learning rate slider
        ivar = Tk.IntVar()
        self.learning_rate_slider_label = Tk.Label(self.sliders_frame, text="Learning Rate")
        self.learning_rate_slider_label.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.learning_rate_slider = Tk.Scale(self.sliders_frame, variable=Tk.DoubleVar(), orient=Tk.HORIZONTAL,
                                             from_=0.001, to_=1, resolution=0.01, bg="#DDDDDD",
                                             activebackground="#FF0000",
                                             highlightcolor="#00FFFF", width=10,
                                             command=lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.set(self.learningRate)
        self.learning_rate_slider.bind("<ButtonRelease-1>", lambda event: self.learning_rate_slider_callback())
        self.learning_rate_slider.grid(row=0, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Sample size percentage slider
        self.sampleSizePerc_slider_label = Tk.Label(self.sliders_frame, text="Sample Size Percentage")
        self.sampleSizePerc_slider_label.grid(row=3, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.sampleSizePerc_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                              from_=10, to_=100, bg="#DDDDDD",
                                              activebackground="#FF0000",
                                              highlightcolor="#00FFFF", width=10)
        self.sampleSizePerc_slider.set(self.sampleSizePerc)
        self.sampleSizePerc_slider.bind("<ButtonRelease-1>",
                                        lambda event: self.sampleSizePerc_slider_callback())
        self.sampleSizePerc_slider.grid(row=3, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Batch size slider
        self.batchSize_slider_label = Tk.Label(self.sliders_frame, text="Batch Size")
        self.batchSize_slider_label.grid(row=4, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.batchSize_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                         from_=10, to_=100, bg="#DDDDDD",
                                         activebackground="#FF0000",
                                         highlightcolor="#00FFFF", width=10)
        self.batchSize_slider.set(self.sampleSizePerc)
        self.batchSize_slider.bind("<ButtonRelease-1>",
                                   lambda event: self.batchSize_slider_callback())
        self.batchSize_slider.grid(row=4, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # Number of iterations slider
        self.iterations_slider_label = Tk.Label(self.sliders_frame, text="Number of Iterations")
        self.iterations_slider_label.grid(row=5, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.iterations_slider = Tk.Scale(self.sliders_frame, variable=Tk.IntVar(), orient=Tk.HORIZONTAL,
                                          from_=1, to_=10, bg="#DDDDDD",
                                          activebackground="#FF0000",
                                          highlightcolor="#00FFFF", width=10)
        self.iterations_slider.set(self.iterations)
        self.iterations_slider.bind("<ButtonRelease-1>",
                                    lambda event: self.iterations_slider_callback())
        self.iterations_slider.grid(row=5, column=1, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # set weights to zero
        self.setWeights_zero_button = Tk.Button(self.buttons_frame,
                                                text="Set Weights Zero", bg="yellow", fg="red",
                                                command=lambda: self.setWeights_zero_button_callback())
        self.setWeights_zero_button.grid(row=1, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        # adjust weights
        self.adjust_weights_button = Tk.Button(self.buttons_frame,
                                               text="Adjust Weights",
                                               bg="yellow", fg="red",
                                               command=lambda: self.adjust_weights_button_callback())
        self.adjust_weights_button.grid(row=2, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)
        self.initialize()

    def initialize(self):
        self.calculate_sample_batch()
        self.weights = self.randomize_weights(2, self.number_of_delayedelements + 1)

    def iterations_slider_callback(self):
        self.iterations = self.iterations_slider.get()
        print self.iterations

    def adjust_weights_button_callback(self):
        #print "adjust weights"
        self.create_inputSamples()
        #print self.batch_p
        #print self.batch_v
        no_of_times = self.iterations
        for i in range(no_of_times):
            self.adaline()
            print i

    def setWeights_zero_button_callback(self):
        self.weights = self.randomize_weights(2,self.number_of_delayedelements+1)
        self.weights.fill(0.0)
        print self.weights
        print "SetWeights"
        print "set weights to zero"

    def number_of_delayedelements_slider_callback(self):
        print "number of delayed elements"
        self.number_of_delayedelements=self.number_of_delayedelements_slider.get()
        print self.number_of_delayedelements
        self.weights=self.randomize_weights(2,self.number_of_delayedelements+1)
    def learning_rate_slider_callback(self):
        self.learningRate = self.learning_rate_slider.get()
        print self.learningRate
    def sampleSizePerc_slider_callback(self):
        self.sampleSizePerc = self.sampleSizePerc_slider.get()
        ssp = self.sampleSizePerc
        self.current_sample_size = (self.totalsize * ssp) / 100
        bsp = self.batchSize
        self.current_batch_size = (self.current_sample_size * bsp) / 100
        print self.current_sample_size
        print self.current_batch_size
        self.weights = self.randomize_weights(2, self.number_of_delayedelements + 1)
        self.calculate_sample_batch()
    def batchSize_slider_callback(self):
        self.batchSize = self.batchSize_slider.get()
        print self.batchSize
        bsp = self.batchSize
        self.current_batch_size = (self.current_sample_size * bsp) / 100
        self.weights = self.randomize_weights(2, self.number_of_delayedelements + 1)
        print self.current_batch_size
    def iterations_slider_callback(self):
        self.iterations = self.iterations_slider.get()
        print self.iterations
    def calculate_sample_batch(self):
        # normalize data
        prize = self.stockdata[:, 0] / 214.0
        volume = self.stockdata[:, 1] / 16337400
        p = [round(i, 2) for i in prize.tolist()]
        v = [round(i, 2) for i in volume.tolist()]
        self.totalsize = self.stockdata.shape[0]
        self.current_sample_size = (self.totalsize * self.sampleSizePerc) / 100
        self.current_batch_size = (self.current_sample_size * self.batchSize) / 100
        self.sample_p = p[:self.current_sample_size]
        self.sample_v = v[:self.current_sample_size]
        print self.sample_p
        print self.sample_v
    def randomize_weights(self,no_rows,no_cols):
        min_initial_weights = -1
        max_initial_weights = 1
        random_weights=np.random.uniform(min_initial_weights, max_initial_weights, (no_rows, no_cols + 1))
        print np.round(random_weights,2)
        return np.round(random_weights,2)


    def create_inputSamples(self):
        a = []
        b=[]
        self.batch_p=[]
        self.batch_v = []
        i = 0
        for j in range(self.current_sample_size):
            i = i + 1
            a.append(self.sample_p[j])
            b.append(self.sample_v[j])
            if i == self.current_batch_size:
                i = 0
                self.batch_p.append(a)
                self.batch_v.append(b)
                a = []
                b=[]
        self.batch_p.append(a)
        self.batch_v.append(b)



    def adaline(self):
        no_batches=len(self.batch_v)
        for i in range(no_batches):
            prize_veclist=self.batch_p[i]
            volume_veclist=self.batch_v[i]
            self.create_delayed(prize_veclist,volume_veclist)

    def create_delayed(self,prize,volume):
        av=[]
        bv = []
        i=0
        delayed_prize=[]
        delayed_volume=[]
        for j in range(len(prize)):
            i = i + 1
            one = 1.0
            av.append(prize[j])
            bv.append(volume[j])

            if i == self.number_of_delayedelements+1:
                i = 0
                av.append(one)
                bv.append(one)
                delayed_prize.append(av)
                delayed_volume.append(bv)
                av = []
                bv = []
        if len(av)<=self.number_of_delayedelements+1:
            for i in range((self.number_of_delayedelements+1)-len(av)):
                av.append(0.0)
                bv.append(0.0)
        av.append(one)
        bv.append(one)
        delayed_prize.append(av)
        delayed_volume.append(bv)
        self.calculate_output(delayed_prize,delayed_volume)

    def calculate_output(self,delayed_prize,delayed_volume):
        print "calculate_output"
        # Creating targets for prize and volume
        tp=[]
        tv=[]
        for i in range(len(delayed_prize)):
            tp.append(delayed_prize[i][0])
            tv.append(delayed_volume[i][0])
        #print tp
        #print tv
        if (len(tp))!=1:
            t_p=tp[0]
            t_v=tv[0]
            del tp[0]
            del tv[0]
            tp.append(t_p)
            tv.append(t_v)
        #print tp
        #print tv
        # Lms algorithm
        err_b_p=[]
        err_b_v=[]
        for j in range(len(delayed_prize)):
            bp=np.array(delayed_prize[j])
            bv=np.array(delayed_volume[j])
            vpmatrix = np.vstack((bp))
            vvmatrix = np.vstack((bv))
            activation_price = self.output(self.weights[0], vpmatrix)
            activation_volume = self.output(self.weights[1], vvmatrix)
            error_p = self.calculateErrors(tp[j], activation_price)
            error_v=self.calculateErrors(tv[j], activation_volume)
            err_b_p.append(error_p)
            err_b_v.append(error_v)

            self.weights[0] = self.calculate_newWeights(error_p, self.weights[0], vpmatrix)
            self.weights[1] = self.calculate_newWeights(error_v, self.weights[1], vvmatrix)
        ev=[max(err_b_v),err_b_v[-1]*err_b_v[-1]]
        ep=[max(err_b_p),err_b_p[-1]*err_b_p[-1]]
        self.error_display(ep,ev)

    def error_display(self,error_p,error_v):


        self.batch_count=+1
        plt.scatter(error_p[0],self.batch_count)
        plt.scatter(error_p[1], self.batch_count)
        plt.scatter(error_v[0], self.batch_count)
        plt.scatter(error_v[1], self.batch_count)
        #plt.scatter(price_err, [2, 4,8,10,5,6])
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.display_frame)
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.grid(row=0, column=0, sticky=Tk.N + Tk.E + Tk.S + Tk.W)


    def output(self, matrix_weight, matrix_a):
        return matrix_weight.dot(matrix_a).item()
        #or i in range(len(delayed_prize)):
    def calculateErrors(self, t, a):
        return round(t-a)

    def calculate_newWeights(self, e, w, p):
        alpha = self.learningRate
        pvector = p.transpose()
        first = w
        sec = 2 * alpha * e * pvector
        return first + sec
        # d=[[0.0] * i for i in range(self.number_of_delayedelements+1+no_of_targets)]
        # print "adaline"
        # #d.append(1.0)

        # no_batches=len(self.batch_v)
        # for i in range(no_batches):
        #     d_vec=[]
        #
        #     vec_list=self.batch_v[i]
        #     for j in range(len(vec_list)):
        #         for k in range(self.number_of_delayedelements+1+no_of_targets):
        #             d[k]=vec_list[j]
        #         print d
        #         d_vec.append(d)
        #
        #     print d_vec







if __name__ == "__main__":
    stockdata = read_csv_as_matrix("stock_data.csv")

    prize=stockdata[:,0]/214.0
    volume=stockdata[:,1]/16337400


    #print v
    main_frame = Tk.Tk()
    main_frame.title("Widrow-Huff learning and adaptive filters")
    main_frame.geometry('820x640')
    ob_nn_gui_2d = ClNNGui2d(main_frame,stockdata)
    main_frame.mainloop()