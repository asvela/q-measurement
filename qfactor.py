# -*- coding: utf-8 -*-
"""


Andreas Svela // 2019
"""

import sys, os
import logging; _log = logging.getLogger(__name__)
import pandas as pd             #dataframes
import numpy as np              #arrays
import matplotlib.pyplot as plt #plotting
import scipy.signal as sig
import scipy.optimize as opt    #fitting
import keyoscacquire.programmes as acq
from matplotlib import gridspec
from datetime import datetime

_filetype = ".csv"

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def read_data(fname, ext=_filetype, usecols=[0, 1], names=['time', 'ch1']):
    df = pd.read_csv(fname+ext, skiprows=[0, 1, 2], usecols=usecols, names=names)
    return [df[col].values for col in names]

def frequency_span_per_sec(scan_freq, peak_to_peak, scaling, calibration):
    """Calculate frequency span per second for the laser in MHz per second:
    scan_freq in Hz,  peak_to_peak in volts,  scaling in mA/volt,
    calibration in MHz/mA"""
    scan_period = 1/(2*scan_freq) #sec, division by two because of triangle wave and hence in practice sweeping double the speed
    current_span = peak_to_peak*scaling #mA
    return current_span*calibration/scan_period #MHz/second

def scale_channel(channel, actual_zero=0, new_min=0, new_max=1, max_func=np.max):
    """Scale the channels, mapping them from actual zeros and max value to 0 to 1"""
    zero = actual_zero if actual_zero is not None else np.min(channel)
    channel = channel - zero
    corrected_max = max_func(channel)
    old_range = corrected_max
    new_range = new_max - new_min
    scaling = new_range/old_range
    return channel*scaling + new_min

moving_average = lambda oneD_array, window: np.convolve(oneD_array, np.ones((window,))/window, mode='valid')

def lor_fit(f, background, amp, f0, linewidth):
    return background - amp/(1 + (f - f0)**2/(linewidth/2)**2)

def read_and_calc_Q(folder, fname, pump_freq, freq_per_sec, subfolder="", truncation_factor=10, showplt=True):
    # read data
    time, ch1, ch2 = read_data(folder+subfolder+fname, usecols=[0,1,2], names=['time', 'ch1', 'ch2'])
    trace = scale_channel(ch1, max_func=np.mean)
    freq = time*freq_per_sec

    # fit the trace
    ## guess on the form (background, amp, f0, linewidth)
    guess = (np.mean(trace), np.max(trace)-np.min(trace), freq[np.argmin(trace)], 0.5)
    bounds = ((0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf))
    ## truncate the trace
    indices = ((guess[2]-truncation_factor*guess[3]) < freq) & (freq < (guess[2]+truncation_factor*guess[3]))
    trunc_freq = freq[indices]
    trunc_trace = trace[indices]
    ## run fit and extract parameters
    optp, _ = opt.curve_fit(lor_fit, trunc_freq, trunc_trace, p0=guess, bounds=bounds)
    background, amp, f0, linewidth = optp
    fitline = lor_fit(trunc_freq, *optp)

    # calculate the Q
    Q = pump_freq/linewidth
    print("Q = %.1fe8" % (Q/1e8))
    Q_str = r"$Q$ = %.1f$\cdot10^8$" % (Q/1e8)
    lw_str = r"$\gamma/2\pi$ = %.0f kHz" % (linewidth*1e3)

    # plot the fit and linewidth
    ax = plt.subplot()
    ax.plot(freq-f0, trace)
    # ax.plot(freq, lor_fit(freq, *guess))
    ax.plot(trunc_freq-f0, fitline)
    ax.set_xlim([-8*linewidth, 8*linewidth])
    ax.set_xlabel("Frequency offset [MHz]")
    ax.set_ylabel("Transmission")
    ax.set_title(fname+": "+Q_str+", "+lw_str)
    ax.hlines(y=amp/2+np.min(fitline), xmin=-linewidth/2, xmax=linewidth/2)
    ax.annotate(Q_str, (-linewidth, background), xycoords='data')
    plt.tight_layout()
    plt.savefig(folder+fname+(" Q%.1fe8.png"%(Q/1e8)))
    if showplt:
        plt.show()
    else:
        plt.close()

def calculate_folder(folder, pump_freq, freq_per_sec):
    for fname in list_fnames(folder):
        print(fname, end=": ")
        read_and_calc_Q(folder, fname, pump_freq, freq_per_sec, showplt=False)

def list_fnames(folder="./"):
    return [fname[:-4] for fname in os.listdir(folder) if '.csv' in fname]


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
##                                   MAIN                                     ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

if __name__ == '__main__':
    # folder = "T:/DATA/Microcombs/Experiments/Near-field/Code/Q-measurement/"
    folder = "./"#scan speed 20Hz/"
    subfolder = "data/"
    fname = sys.argv[1] if len(sys.argv) >= 2 else "q-meas"
    a_type = sys.argv[2] if len(sys.argv) >= 3 else "HRES"

    wavelength = 1550e-9 #m
    pump_freq = 3e8/wavelength/1e6 #MHz
    # scan_freq, calibration = 1007, 8.2
    scan_freq, calibration = 20, 11.2
    freq_per_sec = frequency_span_per_sec(scan_freq=scan_freq, peak_to_peak=2, scaling=10, calibration=calibration)

    if not os.path.exists(folder+subfolder+fname+_filetype):
        print("Obtaining trace..")
        acq.get_single_trace(fname=folder+subfolder+fname, acq_type=a_type)

    print("Calculating Q..")
    read_and_calc_Q(folder, fname, pump_freq, freq_per_sec, subfolder=subfolder)
    # calculate_folder(folder, pump_freq, freq_per_sec)
