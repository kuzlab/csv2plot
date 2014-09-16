#!/usr/bin/python

import argparse

vol_average_default = 1.65

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action = "store_true", default = False)
parser.add_argument("-q", "--quiet", action = "count", default = 0)
parser.add_argument("-hf", "--hoge-fuga", default = "")
parser.add_argument("infilename", default = '', type = str)
parser.add_argument("outextention", default = 'figure.png', type = str)
parser.add_argument("start_point", default = 0, type = int)
parser.add_argument("stop_point", default = 10240, type = int)
parser.add_argument("vol_average", default = vol_average_default, type = float)
parser.add_argument("ave1", default = 10, type = int)
parser.add_argument("ave2", default = 50, type = int)
args = parser.parse_args()

DEBUG_PRINT = 0

if DEBUG_PRINT:
    print(args.infilename, args.outextention, args.vol_average)

import csv

def data_loading_from_cvsfile(filename, data_shift_for_starting_point):
    raw_data = [ v for v in csv.reader(open(filename, "r")) if len(v) != 0]
    if DEBUG_PRINT:
        print(raw_data)
        for x in range(len(raw_data)):
            print(raw_data[x][0])
        print(len(raw_data) )
        print(float(raw_data[3][0]) - data_shift)

    return_data = [0.0] * (len(raw_data) - data_shift_for_starting_point)

    for x in range(len(raw_data) - data_shift_for_starting_point):
        return_data[x] = float(raw_data[x + data_shift_for_starting_point][0])
    
    return return_data, raw_data

    
data_shift = 2
f_data, csv_data = data_loading_from_cvsfile(args.infilename, data_shift)

if DEBUG_PRINT:
    print(f_data)

######### DATA LOADING OK #############

if len(f_data) < args.stop_point:
    args.stop_point = len(f_data)

data_length = args.stop_point - args.start_point

average_center = 1.65

plot_data = [0.0] * (data_length)

#for x in range(data_length):
for x in range(data_length):
    plot_data[x] = abs(f_data[x + args.start_point] - 1.65)

print(plot_data)

import numpy as np
#import matplotlib.pyplot as plt

x_num = np.arange(args.start_point, args.stop_point, 1)

#plt.plot(x, f_data)

import pylab
#pylab.plot(x_num, plot_data, label="abs")


def plot_average(ave_num):
    ave1 = int(ave_num)
    ave1_data = [0.0] * (data_length - ave1 +1)
    for j in range(0, data_length - ave1):
        temp = [0.0] * (ave1)
        for i in range(0, ave1):
#           print(f_data[i + j])
            temp[i] = plot_data[i + j]
#       print(temp)
        ave1_data[j] = np.average(temp)
    label = str(ave1) + "points average"
    x_num = np.arange(args.start_point, args.stop_point - ave1 + 1, 1)
    pylab.xlim(args.start_point, args.stop_point)    
    pylab.plot(x_num, ave1_data, label=label)
    pylab.legend()
#print(ave1_data)


#pylab.subplot(3, 1, 1)
#plot_average(args.ave1)
#pylab.subplot(3, 1, 2)
#plot_average(args.ave2)
#pylab.subplot(3, 1, 3)
ave = 1
ave_data = [0.0] * (data_length - ave +1)
for j in range(0, data_length - ave):
    temp = [0.0] * (ave)
    for i in range(0, ave):
        temp[i] = plot_data[i + j] * plot_data[i + j]
    ave_data[j] = np.average(temp)
x_num = np.arange(args.start_point, args.stop_point - ave + 1, 1)
label = str(ave) + " pt ave -> pow"

if 0:
    pylab.subplot(3, 1, 1)
    pylab.xlim(args.start_point, args.stop_point)
    pylab.plot(x_num, ave_data, label=label)
    pylab.legend()

import scipy.signal

def plot_lpf(N, Fc, Fs, x):
    h=scipy.signal.firwin( numtaps=N, cutoff=40, nyq=Fs/2)
    y=scipy.signal.lfilter( h, 1.0, ave_data)
    label = "LPF(N=" + str(N) + ',Fc=' + str(Fc) + ",Fs=" + str(Fs) +")"
#    pylab.plot(x_num, y, label=label)
    pylab.ylim(0.05, 2)
    pylab.xlim(args.start_point, args.stop_point)
    pylab.semilogy(x, y, label=label)
    pylab.grid(True)
    pylab.legend()
    return y
    

N=100    # tap num
Fc=50   # cut off
Fs=3000 # sampling

pylab.subplot(3, 1, 1)
lpf_data = plot_lpf(N, Fc, Fs, x_num)


import time
count = time.time()
t = time.localtime()
t_char = str(t.tm_year) + "-" + str(t.tm_mon) + "-" + str(t.tm_mday) + "_" + str(t.tm_hour) + "-" + str(t.tm_min) + "-" + str(t.tm_sec)

period = csv_data[0][0]
period = period[7:]

div = float(period) # sec
print(div)
#len(data) - data_shift /10 = div

sample_div = div/(len(f_data)/10)
sound_speed = 1500 #m/sec

x_time = x_num / sample_div         # sec
x_distance = x_time * sound_speed / 2  # meter


pylab.subplot(3, 1, 2)
pylab.xlabel("time[sec]")
label="LPF vs time"
pylab.plot(x_time, lpf_data, label=label)
pylab.legend()

pylab.subplot(3, 1, 3)
pylab.xlabel("distance[m]")
label="LPF vs distance"
pylab.plot(x_distance, lpf_data, label=label)
pylab.legend()


from scipy import fftpack
sample_freq = fftpack.fftfreq(len(ave_data), 0.6)
sample_sig = fftpack.fft(ave_data)
sample_pow = abs(sample_sig)

if 0:
    pylab.subplot(3, 1, 2)
    label="fft"
    pylab.xlim(xmin=0)
    #pylab.plot(sample_freq, sample_pow, label=label)
    pylab.semilogy(sample_freq, sample_pow, label=label)
    pylab.grid(True)
    pylab.legend()

if 0:
    pylab.subplot(3, 1, 3)
    label="LPF->fft"
    pylab.xlim(xmin=0)
    sample_sig_lpf = fftpack.fft(lpf_data)
    sample_pow_lpf = abs(sample_sig_lpf)
    pylab.semilogy(sample_freq, sample_pow_lpf, label=label)
    pylab.grid(True)
    pylab.xlabel("Freq")
    pylab.legend()


outfile = args.infilename + "_" + args.outextention + "_" + t_char + ".png"
pylab.savefig(outfile)
pylab.show()


#print(data[0])
#print(data[1])
#print(data[2][0])
#print(data[3][0])
#print(data[4][0])
#print(data[5][0])



#data = ['1', '2.0', '3', 'data']

#test = data[0].strip().isdigit()
#print(test)
#test = data[1].strip().isdigit()
#print(test)
#test = data[2].strip().isdigit()
#print(test)
#test = data[3].strip().isdigit()
#print(test)

#def is_number(i):
#    for v in range(i):
#        print i[v].strip().isdigit()

#print(is_number(data))


#data = filter(lambda x:False not in [v.strip().isdigit() for v in x], data)
#print(data)


#data = [ v for v in csv_obj]
#data_conved = [[int(elm) for elm in v] for v in data]
#print(data_conved)



#csv_obj = csv.render(open(args.infilename, "r"))
#data = [ v for v in csv_obj]

#data = [mol for mol in [[int(nucl) for nucl in atom if len(nucl) != 0 and nucl.strip().isdigit()] for atom in csv.reader(open("bad.csv", "r"))] if len(mol) != 0]
#print(data)


#>>>data = [ v for v in csv.reader(open("bad.csv", "r")) if len(v) != 0]
#>>> print(data)


#data = filter(lambda x:False not in [v.strip().isdigit() for v in x], data)
#>>> print(data)

#>>> data = map(lambda x:[ int(v) for v in x], data)
#>>> print(data)
