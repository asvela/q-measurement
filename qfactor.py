# -*- coding: utf-8 -*-
"""
Q-factor measurement tool and calculator

Run with option -h to learn more.

Andreas Svela // 2019
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import keyoscacquire.oscacq as oscacq

_filetype = ".csv"
_oscilloscope_visa_address = "" # leave empty to use default address for keyoscacquire
                                # (the default address can be set by changing the file
                                #  whose path is given by running 'path_of_config' in cmd)


## Mathematical functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def moving_average(oneD_array, window):
    return np.convolve(oneD_array, np.ones((window,))/window, mode='valid')

def lorentzian(f, background, amp, f0, linewidth):
    return background - amp/(1 + (f - f0)**2/(linewidth/2)**2)

def three_lorentzians(f, delta_f, background, amp_low, amp_0, amp_high, f0, linewidth):
    amplitudes = (amp_low, amp_0, amp_high)
    f0s = (f0-delta_f, f0, f0+delta_f)
    lorentzians = np.array([amp/(1 + (f - f0)**2/(linewidth/2)**2) for amp, f0 in zip(amplitudes, f0s)])
    lorentzians = np.sum(lorentzians, axis=0)
    return background - lorentzians


## File IO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def read_data(fname, ext=_filetype, usecols=[0, 1], names=['time', 'ch1']):
    """Read trace data"""
    if fname[-4:] == ".csv":
        fname = fname[:-4]
    df = pd.read_csv(fname+ext, skiprows=[0, 1, 2, 3], usecols=usecols, names=names)
    return [df[col].values for col in names]

def read_zero(fname):
    """Get mean of vacuum trace for photo diode calibration"""
    return np.mean(read_data(fname, usecols=[1], names=['ch1']))

def list_fnames(folder="./"):
    """List of all files ending with .csv in a folder"""
    return [fname[:-4] for fname in os.listdir(folder) if '.csv' in fname]


##  Trace preparation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def scale_trace(trace, PD_zero=0, new_min=0, new_max=1, max_func=np.max):
    """Scale the channels, mapping them from actual zeros and max value to 0 to 1"""
    zero = PD_zero if PD_zero is not None else np.min(trace)
    trace = trace - zero
    corrected_max = max_func(trace)
    old_range = corrected_max
    new_range = new_max - new_min
    scaling = new_range/old_range
    return trace*scaling + new_min

def make_guess(freqaxis, trace, linewidth_guess=10):
    """Make a guess for the fit parameteres based on the trace data,
    and set the bounds to be only positive numbers background, amplitude,
    and linewidth"""
    # (background, amp, f0, linewidth)
    guess = (np.mean(trace), np.max(trace)-np.min(trace), freqaxis[np.argmin(trace)], linewidth_guess)
    bounds = ((0, 0, -np.inf, 0), (np.inf, np.inf, np.inf, np.inf))
    return guess, bounds

def make_guess_three_lorentzians(time, trace):
    """Make a guess for the fit parameteres based on the trace data"""
    # delta_f, background, amp_low, amp_0, amp_high, f0, linewidth
    guess = (np.ptp(time)/8, np.mean(trace), np.ptp(trace)/2, np.ptp(trace),
             np.ptp(trace)/2, time[np.argmin(trace)], np.ptp(time)/20)
    bounds = ((           0,      0,      0,      0,       0, -np.inf,     0),
              (np.ptp(time), np.inf, np.inf, np.inf,  np.inf,  np.inf, np.inf))
    return guess, bounds

def truncate_trace(freqaxis, trace, guess, linewidth_multiple_window=20):
    """Truncate the trace to the part that is within the window defined by
    linewidth_multiple_window*(guessed linewidth) to each side of guessed
    centre frequency"""
    indices = (((guess[2]-linewidth_multiple_window*guess[3]) < freqaxis) &
                (freqaxis < (guess[2]+linewidth_multiple_window*guess[3])))
    trunc_freqaxis = freqaxis[indices]
    trunc_trace = trace[indices]
    return trunc_freqaxis, trunc_trace

## Frequency calibration ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def calc_frequency_span_per_sec(scan_freq, peak_to_peak, scaling, calibration):
    """Calculate frequency span per second for the laser in MHz per second:
    scan_freq [Hz]
    peak_to_peak [Vpp]
    scaling in [mA/V]
    calibration [MHz/mA]"""
    scan_period = 1/(2*scan_freq) #sec
    # (division by two because of triangle wave and hence in practice
    #  sweeping double the speed)
    current_span = peak_to_peak*scaling #mA
    return current_span*calibration/scan_period #MHz/second

def fit_frequency_span_per_sec(fname_sidebands_trace, sidebands_freq,
                               PD_zero=None, showplt=True):
    """Load file of a resonance with sidebands of known separation to obtain
    a frequency axis calibraion in MHz per second"""
    time, ch1 = read_data(fname_sidebands_trace)
    trace = scale_trace(ch1, max_func=np.mean, PD_zero=PD_zero)
    guess, bounds = make_guess_three_lorentzians(time, trace)
    guess_line = three_lorentzians(time, *guess)
    fig, ax = plt.subplots()
    ax.set_xlabel("Time offset [s]")
    ax.set_ylabel("Transmission")
    print("Fitting trace with sidebands.. ", end="")
    try:
        fit_params, _ = opt.curve_fit(three_lorentzians, time, trace,
                                      p0=guess, bounds=bounds)
        print("done")
    except RuntimeError as e:
        ax.plot(time, trace, '.', alpha=0.5, label="data")
        ax.plot(time, guess_line, c="C7", ls="dashed", label="guess")
        fig.suptitle("Failed to fit, guess shown in grey")
        plt.show()
        raise
    delta_t = fit_params[0]
    t0 = fit_params[-2]
    calibration = sidebands_freq/delta_t # MHz/sec
    print(f"The calibration is set to {calibration:,.3f} MHz/sec")
    ax.plot(time-t0, trace, '.', alpha=0.5, label="data")
    ax.plot(time-t0, guess_line, c="C7", ls="dashed", label="guess")
    fitline = three_lorentzians(time, *fit_params)
    ax.plot(time-t0, fitline, label="fit")
    ax.legend()
    fig.suptitle(f"Three Lorentzian fit gave laser sweep calibration {calibration:,.0f} MHz/sec")
    if showplt:
        plt.show()
    if delta_t == 0:
        raise RuntimeError("Got delta_t = 0 from fit of three Lorentzians")
    return calibration


## Now for the Q-measurement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

class QFactorMeasurement:
    pump_freq = 193414489 # MHz, corresponds to 1550nm
    freq_calibrated = False

    def __init__(self, folder="./", PD_zero=None, frequency_span_per_sec=None,
                 acq_type='HRES'):
        self.folder = folder
        self.acq_type = acq_type
        self.set_PD_zero(PD_zero)
        self.set_frequency_span_per_sec(frequency_span_per_sec)

    def set_PD_zero(self, PD_zero=None):
        """Set the vacuum input voltage level by either
        * specifying a float;
        * using a string of a trace file where the first channel is vacuum; or
        * `None` to acquire a vacuum trace and use that"""
        if isinstance(PD_zero, (float, int)):
            self.PD_zero = PD_zero
            print(f"PD value for vacuum is set to {self.PD_zero*1000:.2f}mV")
        elif isinstance(PD_zero, str):
            self.PD_zero = read_zero(PD_zero)
            print(f"PD value for vacuum is set by file '{PD_zero}' to {self.PD_zero*1000:.2f}mV")
        else:
            print("Looks like you should get a vacuum trace (no light on the PD).")
            print("If you don't, the minimum of the resonance trace will be used as zero.")
            acquired = self.acquire(specify=" a vacuum trace")
            if acquired:
                self.PD_zero = read_zero(acquired)
            else:
                print("(!) Warning no photodiode zero is set, minimum will be used.")
                self.PD_zero = None

    def set_frequency_span_per_sec(self, frequency_span_per_sec=None):
        """Set the time to frequency calibration by either
        * specifying a float for MHz swept per second;
        * using a string of a trace file where the first channel has a
          resonace with known sideband separation; or
        * `None` to acquire a trace with sidebands and use that to find
          the calibration"""
        if isinstance(frequency_span_per_sec, (float, int)):
            self.freq_calibrated = True
            self.frequency_span_per_sec = frequency_span_per_sec
            print(f"Frequency calibration is set to {frequency_span_per_sec:,.0f} MHz/sec")
            return
        elif isinstance(frequency_span_per_sec, str):
            fname = frequency_span_per_sec
            print(f"Loading '{fname}' for calibrating frequency axis..")
        else:
            print("Looks like you need to get a trace with some sidebands on it")
            fname = self.acquire(" a trace with sidebands")
        if fname:
            sidebands_freq = float(input("At what frequency are the sidebands? [MHz] "))
            self.frequency_span_per_sec = fit_frequency_span_per_sec(fname, sidebands_freq)
        else:
            print(f"I could't work with the filename '{fname}', sorry")

    def acquire(self, specify="", acq_type=None):
        """Ask for a filename to use and acquire a trace with the oscilloscope"""
        fname = input(f"When ready to acquire{specify}, type filename for the "
                       "trace (no extension)\nand enter (or 'n' to cancel):\n")
        if not fname.lower() == 'n':
            fname = self.folder+fname
            # optional override of acq_type
            acq_type = self.acq_type if acq_type is None else acq_type
            if _oscilloscope_visa_address:
                scope = oscacq.Oscilloscope(_oscilloscope_visa_address)
            else:
                scope = oscacq.Oscilloscope()
            scope.set_options_get_trace_save(fname, _filetype, acq_type=acq_type)
            return fname
        else:
            return False

    def fit(self, print_Q=True):
        """"Fit a Q-factor trace with the resonance only, no sidebands"""
        self.guess, bounds = make_guess(self.freqaxis, self.trace)
        self.trunc_freq, self.trunc_trace = truncate_trace(self.freqaxis, self.trace, self.guess)
        self.guess_line = lorentzian(self.trunc_freq, *self.guess)
        print("Fitting trace.. ", end="")
        try:
            self.fit_params, _ = opt.curve_fit(lorentzian, self.trunc_freq, self.trunc_trace,
                                               p0=self.guess, bounds=bounds)
            print("done")
        except RuntimeError as e:
            fig, ax = plt.subplots()
            ax.plot(self.freqaxis, self.trace, '.')
            ax.plot(self.trunc_freq, self.guess_line, c="C7", ls="dashed")
            fig.suptitle("Failed to fit, guess shown in grey")
            plt.show()
            print("\n", e)
        self.background, self.amp, self.f0, self.linewidth = self.fit_params
        self.Q = self.pump_freq/self.linewidth
        self.Q0 = 2*self.Q/(1+np.sqrt(1-self.amp))
        if print_Q:
            print(f"Q   = {self.Q/1e8:.3f}e8\n"
                  f"Q_0 = {self.Q0/1e8:.3f}e8")

    def plot(self, plot_guess=False, showplt=True):
        """Plot the data, fit, and linewidth"""
        Q_str = rf"$Q$ = {self.Q/1e8:.3f}$\cdot10^8$"
        Q0_str = rf"$Q_0$ = {self.Q0/1e8:.1f}$\cdot10^8$"
        lw_str = rf"$2\gamma/2\pi$ = {self.linewidth:.2f} MHz"
        fitline = lorentzian(self.trunc_freq, *self.fit_params)
        fig, ax = plt.subplots()
        ax.plot(self.freqaxis-self.f0, self.trace, '.', alpha=0.5, label="data")
        ax.plot(self.trunc_freq-self.f0, fitline, label="fit")
        ax.set_xlim([-8*self.linewidth, 8*self.linewidth])
        ax.set_xlabel("Frequency offset [MHz]")
        ax.set_ylabel("Transmission")
        ax.legend()
        ax.set_title(f"{self.fname}\n{Q_str}, {lw_str}")
        ax.hlines(y=self.amp/2+np.min(fitline), xmin=-self.linewidth/2, xmax=self.linewidth/2, color="k")
        ax.annotate(Q_str, (-self.linewidth, self.background), xycoords='data')
        ax.annotate(Q0_str, (-self.linewidth, self.background-self.amp*1.05), xycoords='data')
        plt.tight_layout()
        plt.savefig(f"{self.folder}{self.fname} Q{self.Q/1e8:.3f}e8.png")
        # plot guess after saving
        if plot_guess:
            ax.plot(self.trunc_freq-self.f0, self.guess_line, c="C7", ls="dashed", label="guess")
            ax.legend()
        if showplt:
            plt.show()
        else:
            plt.close()

    def acquire_and_calc_Q(self, showplt=True, acq_type=None):
        """Acquire a trace of the resonance and then calculate the Q-factor"""
        fname = self.acquire(" trace for calculating the Q", acq_type)
        if fname:
            self.read_and_calc_Q(fname)
        else:
            print("Trace not successfully obtained")

    def read_and_calc_Q(self, fname, showplt=True):
        """Read an existing file and calculate its Q-factor"""
        self.fname = fname
        time, ch1 = read_data(self.folder+fname)
        self.trace = scale_trace(ch1, max_func=np.mean, PD_zero=self.PD_zero)
        self.freqaxis = time*self.frequency_span_per_sec
        self.fit()
        self.plot()

# def calculate_folder(folder, PD_zero):
#     for fname in list_fnames(folder+subfolder):
#         print(fname, end=": ")
#         read_and_calc_Q(folder, fname, pump_freq, freq_per_sec, showplt=False, subfolder=subfolder, PD_zero=PD_zero)


## MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def main():
    parser = argparse.ArgumentParser(description='Measure and calculate Q-factor of a resonance')
    parser.add_argument('-r', '--resonance-filename', type=str, default=None,
                        help=("Filename of a trace sweepting the resonance, "
                              "or leave empty to acquire a trace."))
    parser.add_argument('-v', '--vacuum-filename', type=str, default=None,
                        help=("Filename of a trace with no light input on the PD, "
                              "or leave empty to acquire one."))
    parser.add_argument('-c', '--frequency_calibration', type=str, default=None,
                        help=("Float giving the laser sweep rate (MHz/sec), "
                              "or the filename of a trace with sidebands (will ask for sideband frequency later), "
                              "or leave empty to acquire a trace with sidebands."))
    parser.add_argument('-f', '--folder', type=str, default="./",
                        help="Select a folder if different from the folder where the script is exectuted")
    args = parser.parse_args()
    print("Hello! Let's get started with this measurement!")
    # Test if float given instead of filename for calibration
    try:
        args.frequency_calibration = float(args.frequency_calibration)
    except (ValueError, TypeError): # VE triggered by non-permissible characters
        pass                        # TE triggered by `None`
    # Spawn class with the settings given
    qfm = QFactorMeasurement(folder=args.folder, PD_zero=args.vacuum_filename,
                             frequency_span_per_sec=args.frequency_calibration)
    # Check if filename was given
    if args.resonance_filename:
        qfm.read_and_calc_Q(args.resonance_filename)
    else:
        # When no filename given, acquire and calculate
        qfm.acquire_and_calc_Q()
    print("Good bye, you will be missed.")


if __name__ == '__main__':
    main()
