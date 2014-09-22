#!/usr/bin/python
import argparse
import csv
import numpy as np
import pylab
from scipy import fftpack
import scipy.signal
import time

###### settings ##########
DEBUG_PRINT = 0

plot_mode = 0
# plot_mode = 0 : raw data -> abs(v-v0) -> ave1, ave2
# plot_mode = 1 : abs(v-v0) -> average -> LPF(ave1) -> distance
# plot_mode = 2 : LPF -> short period FFT -> all sum -> time domain
# plot_mode = 3 : short period FFT -> limited freq sum(BPF) -> time domain

total_plot_num = 4
plot_count = 1

vol_average_default = 1.65  # 3.3V/2 = 1.65V

data_shift = 2  # beggining of 2 rows skip (no data)

# LPF parameters
N=100    # tap num
Fc=50   # cut off
Fs=3000 # sampling

###############################


############################################################
### definitions
############################################################

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

def return_average(ave_num, data):
    ave1 = int(ave_num)
    length = args.stop_point - args.start_point
    ave1_data = [0.0] * (length - ave1 +1)
    for j in range(0, length - ave1):
        temp = [0.0] * (ave1)
        for i in range(0, ave1):
            temp[i] = data[i + j]
        ave1_data[j] = np.average(temp)
    if DEBUG_PRINT:
        print(ave1_data)
    return ave1_data
        
def subplot_normal(x, y, plot_num_total, plot_order, labels):
    pylab.subplot(int(plot_num_total), 1, int(plot_order))
    pylab.xlim(x[0], x[len(x) - 1]) 
    pylab.plot(x_num, y, label=labels[0])
    pylab.xlabel(labels[1])
    pylab.ylabel(labels[2])
    pylab.legend(loc="lower right")
    pylab.grid(True)

def subplot_log(x, y, plot_num_total, plot_order, labels, y_minmax):
    pylab.subplot(int(plot_num_total), 1, int(plot_order))
    pylab.xlim(x[0], x[len(x) - 1])    
    pylab.semilogy(x, y, label=labels[0])
    pylab.xlabel(labels[1])
    pylab.ylabel(labels[2])
    pylab.ylim(float(y_minmax[0]), float(y_minmax[1]))
    pylab.legend(loc="lower right")
    pylab.grid(True)
    
def return_lpf(N, Fc, Fs, x, data):
#    h=scipy.signal.firwin( numtaps=N, cutoff=40, nyq=Fs/2)
    h=scipy.signal.firwin( numtaps=N, cutoff=Fc, nyq=Fs/2)    
    y=scipy.signal.lfilter( h, 1.0, data)
    return y
    
def return_hpf(N, Fc, Fs, x, data):
    #h=scipy.signal.firwin( numtaps=N, cutoff=Fc, nyq=Fs/2)    
    #y=scipy.signal.lfilter( h, 1.0, data)
    h = scipy.signal.firwin(numtaps=N, cutoff=Fc, nyq=Fs/2, pass_zero=False)
    y= scipy.signal.lfilter( h, 1.0, data)    
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y
    
def power_sum_of_fft(data, min, max):
    sample_freq = fftpack.fftfreq(len(data), 0.6)
    sample_sig = fftpack.fft(data)
    sample_pow = abs(sample_sig)
#    print(sample_freq)
    sum = 0.0
    for i in range(len(sample_pow)):
        if sample_freq[i] >= min:
            if sample_freq[i] <= max:
                sum = sum + sample_pow[i]
    return sum
    
############################################################
### definitions end
############################################################

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbose", action = "store_true", default = False)
parser.add_argument("-q", "--quiet", action = "count", default = 0)
parser.add_argument("-hf", "--hoge-fuga", default = "")
parser.add_argument("infilename", default = '', type = str)
parser.add_argument("outextention", default = 'figure.png', type = str)
parser.add_argument("plot_mode", default = 0, type = int)
parser.add_argument("start_point", default = 0, type = int)
parser.add_argument("stop_point", default = 10240, type = int)
parser.add_argument("vol_average", default = vol_average_default, type = float)
parser.add_argument("ave1", default = 10, type = int)
parser.add_argument("ave2", default = 50, type = int)
parser.add_argument("sfft_data_size", default = 50, type = int)
parser.add_argument("sfft_sum_freq_min", default = 0.0, type = float)
parser.add_argument("sfft_sum_freq_max", default = 1.0, type = float)
args = parser.parse_args()

plot_mode = args.plot_mode

if DEBUG_PRINT:
    print(args.infilename, args.outextention, args.vol_average)
    
f_data, csv_data = data_loading_from_cvsfile(args.infilename, data_shift)

if DEBUG_PRINT:
    print(f_data)
    
######### DATA LOADING OK #############

if len(f_data) < args.stop_point:
    args.stop_point = len(f_data)

data_length = args.stop_point - args.start_point

crop_f_data = [0.0] * (data_length)
for i in range(data_length):
    crop_f_data[i] = f_data[i + args.start_point]

if plot_mode == 0:
    temp = return_average(1, crop_f_data)
    labels = ["raw data", "num", "voltage"]
    x_num = np.arange(args.start_point, args.stop_point, 1)
    subplot_normal(x_num, temp, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1

average_center = float(args.vol_average)

plot_data = [0.0] * (data_length)
for x in range(data_length):
    plot_data[x] = abs(f_data[x + args.start_point] - average_center)

x_num = np.arange(args.start_point, args.stop_point, 1)


if DEBUG_PRINT:
    print(plot_data)

if plot_mode == 0:
    temp = return_average(1, plot_data)    
    labels = ["abs(v-v0)", "num", "voltage"]
    x_num = np.arange(args.start_point, args.stop_point, 1)
    subplot_normal(x_num, temp, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1
    temp = return_average(args.ave1, plot_data)
    labels = [str(args.ave1) + "pt ave", "num", "voltage"]
    x_num = np.arange(args.start_point, args.stop_point - args.ave1 + 1, 1)
    subplot_normal(x_num, temp, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1
    temp = return_average(args.ave2, plot_data)
    labels = [str(args.ave2) + "pt ave", "num", "voltage"]
    x_num = np.arange(args.start_point, args.stop_point - args.ave2 + 1, 1)
    subplot_normal(x_num, temp, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1
    
if plot_mode == 3:
    temp = return_average(1, plot_data)    
    labels = ["abs(v-v0)", "num", "voltage"]
    x_num = np.arange(args.start_point, args.stop_point, 1)
    subplot_normal(x_num, temp, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1

if 0:
    ave = 1
    ave_data = [0.0] * (data_length - ave +1)
    for j in range(0, data_length - ave):
        temp = [0.0] * (ave)
        for i in range(0, ave):
            temp[i] = plot_data[i + j] * plot_data[i + j]
        ave_data[j] = np.average(temp)
    x_num = np.arange(args.start_point, args.stop_point - ave + 1, 1)

if plot_mode == 1:
    pow_data = return_average(1, map(lambda x:plot_data[x] * plot_data[x], range(data_length)))
    x_num = np.arange(args.start_point, args.stop_point, 1)
    labels = [str(1) + " pt ave -> pow", "num", "pow"]
    y_min_max = [0.001, 2]
#subplot_log(x_num, pow_data, total_plot_num, plot_count, labels, y_min_max)
#plot_count = plot_count + 1

if 0:
    pylab.subplot(3, 1, 1)
    pylab.xlim(args.start_point, args.stop_point)
    pylab.plot(x_num, ave_data, label=label)
    pylab.legend()

if plot_mode == 1:
    x_num = np.arange(args.start_point, args.stop_point, 1)
    labels = ["LPF(N=" + str(N) + ',Fc=' + str(Fc) + ",Fs=" + str(Fs) +")", "num", "pow"]
    lpf_data = return_lpf(N, Fc, Fs, x_num, pow_data)
    y_min_max = [0.05, 2]
    subplot_log(x_num, lpf_data, total_plot_num, plot_count, labels, y_min_max)
    plot_count = plot_count + 1

period = csv_data[0][0]
period = period[7:]

div = float(period) # sec

if DEBUG_PRINT:
    print("div = ", div) #len(data) - data_shift /10 = div

sample_div = div/(len(f_data)/10)
sound_speed = 1500 #m/sec

if DEBUG_PRINT:
    print("sample = ", sample_div)

x_time = x_num * sample_div         # sec
x_distance = x_time * sound_speed * 100 / 2  # cm

if 0:
    pylab.subplot(3, 1, 2)
    pylab.xlabel("time[sec]")
    label="LPF vs time"
    pylab.plot(x_time, lpf_data, label=label)
    pylab.legend()

if plot_mode == 1:
    labels = ["pow -> abs(v-v0) -> LPF vs distance", "distance[cm]", "pow"]
    y_min_max = [0.05, 2]
    subplot_log(x_distance, lpf_data, total_plot_num, plot_count, labels, y_min_max)
    plot_count = plot_count + 1

if 0:
    sample_freq = fftpack.fftfreq(len(pow_data), 0.6)
    sample_sig = fftpack.fft(pow_data)
    sample_pow = abs(sample_sig)

    pylab.subplot(3, 1, 2)
    label="fft"
    pylab.xlim(xmin=0)
    pylab.semilogy(sample_freq, sample_pow, label=label)
    pylab.grid(True)
    pylab.legend()

    pylab.subplot(total_plot_num, 1, plot_count)
    plot_count = plot_count + 1
    label="LPF->fft"
    pylab.xlim(xmin=0)
    sample_sig_lpf = fftpack.fft(lpf_data)
    sample_pow_lpf = abs(sample_sig_lpf)
    pylab.semilogy(sample_freq, sample_pow_lpf, label=label)
    pylab.grid(True)
    pylab.xlabel("Freq")
    pylab.legend()
    
    
    
##########################################################
#######     2nd idea
#######   raw data -> LPF -> HPF -> distance
##########################################################
if plot_mode == 2:
    x_num = np.arange(args.start_point, args.stop_point, 1)
    ave1_f_data = return_average(1, f_data)

    labels = ["crop data", "num", "voltage"]
    subplot_normal(x_num, ave1_f_data, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1

    # LPF parameters
    N=100    # tap num
    #min_dimension = 0.001 # do not care under 1mm object
    min_dimension = 0.010 # do not care under 10mm
    Fc = sound_speed / (min_dimension/2)   # cut off frequency
    Fs= 1.0 / sample_div    # sampling frequency

    lpf_f_data = return_lpf(N, Fc, Fs, x_num, ave1_f_data)
    labels = ["LPF(N=" + str(N) + ',Fc=' + str(Fc/1000000) + "MHz,Fs=" + str(Fs/1000000) +"MHz)", "num", "voltage"]
    #y_min_max = [0.05, 2]
    subplot_normal(x_num, lpf_f_data, total_plot_num, plot_count, labels)#, y_min_max)
    plot_count = plot_count + 1


## HPF disable
if 0:
    max_dimension = 0.10 # do not care over 10cm object
    Fc_low = sound_speed / (max_dimension/2)   # cut off frequency
    #hpf_lpf_f_data = return_hpf(N, Fc, Fs, x_num, lpf_f_data)

    #Fc_low = 1*1000*1000/100
    Fc_low = 10
    Fc = Fs/2

    #hpf_lpf_f_data = butter_bandpass_filter(ave1_f_data, Fc_low, Fc, Fs, order=5)
    hpf_lpf_f_data = butter_bandpass_filter(lpf_f_data, Fc_low, Fc, Fs, order=5)

    for x in range(len(hpf_lpf_f_data)):
        if hpf_lpf_f_data[x] > 3:
            hpf_lpf_f_data[x] = 3

    labels = ["BPF(N=" + str(N) + ',F1=' + str(Fc_low/1000) + 'k,F2=' + str(Fc/1000) + "k)", "num", "voltage"]
    subplot_normal(x_num, hpf_lpf_f_data, total_plot_num, plot_count, labels)
    plot_count = plot_count + 1


if plot_mode == 2:
    ### try short period FFT + power sumation
    fft_period = args.sfft_data_size
    #fft_period = int(len(f_data) * Fc / Fs)

    fft_sum_data = [0.0] * (data_length - fft_period + 1)
    temp = [0.0] * (fft_period)

    for i in range(len(lpf_f_data) - fft_period + 1):
        for j in range(fft_period):
            temp[j] = lpf_f_data[i + j]
        fft_sum_data[i] = power_sum_of_fft(temp, 0, 1)
    
    x_num = np.arange(args.start_point, args.stop_point - fft_period + 1, 1)
    pylab.subplot(total_plot_num, 1, plot_count)
    labels = ["LPF->" + str(fft_period) + "sampleFFT->t_dom", "num", "pow sum"]
    subplot_normal(x_num, fft_sum_data, total_plot_num, plot_count, labels)
    pylab.grid(True)


if plot_mode == 3:  # plot_mode = 3 : short period FFT -> limited freq sum(BPF) -> time domain
    fft_period = args.sfft_data_size
    fft_sum_data = [0.0] * (data_length - fft_period + 1)
    temp = [0.0] * (fft_period)
    fft_range_min = args.sfft_sum_freq_min
    fft_range_max = args.sfft_sum_freq_max
    for i in range(len(crop_f_data) - fft_period + 1):
        for j in range(fft_period):
            temp[j] = crop_f_data[i + j]
        fft_sum_data[i] = power_sum_of_fft(temp, fft_range_min, fft_range_max)
    x_num = np.arange(args.start_point, args.stop_point - fft_period + 1, 1)        
    pylab.subplot(total_plot_num, 1, plot_count)
    labels = ["sfft(" + str(fft_period) + ", " + str(fft_range_min) + "-" + str(fft_range_max) + ")", "num", "pow sum"]
    subplot_normal(x_num, fft_sum_data, total_plot_num, plot_count, labels)
    pylab.grid(True)
    plot_count = plot_count + 1
    pylab.subplot(total_plot_num, 1, plot_count)    
    labels = ["sfft(" + str(fft_period) + ", " + str(fft_range_min) + "-" + str(fft_range_max) + ")", "distance(cm)", "pow sum"]
    x_distance_fft = [0.0] * (data_length - fft_period)
    for i in (range(data_length - fft_period)):
        x_distance_fft[i] = x_distance[i]
    print(x_distance_fft)
    subplot_normal(x_distance_fft, fft_sum_data, total_plot_num, plot_count, labels)
    pylab.grid(True)
    plot_count = plot_count + 1

count = time.time()
t = time.localtime()
t_char = str(t.tm_year) + "-" + str(t.tm_mon) + "-" + str(t.tm_mday) + "_" + str(t.tm_hour) + "-" + str(t.tm_min) + "-" + str(t.tm_sec)

outfile = args.infilename + "_" + args.outextention + "_" + t_char + ".png"
pylab.savefig(outfile)
pylab.show()



