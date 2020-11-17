# -*- coding: utf-8 -*-
"""


Andreas Svela // 2019
"""

import pandas as pd             #dataframes
import numpy as np              #arrays
import matplotlib.pyplot as plt #plotting
import scipy.signal as sig
from matplotlib import gridspec
from datetime import datetime

_filetype = ".csv"


## Load data and calculate spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

def read_data(fname, ext=_filetype, usecols=[0, 1], names=['time', 'ch1']):
    df = pd.read_csv(fname+ext, skiprows=[0, 1, 2], usecols=usecols, names=names)
    return [df[col].values for col in names]

## Interactive timeseries truncation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

class SetTimeseriesWindowing():
    """Interactive plot:
      * Shows original timeseries, left (right) click sets lower (upper)
        cut-off for the truncated timeseries.
      * Fine adjustment possible in separate smaller plots for lower and
        upper cut-off (period_margin sets the number of periods displayed
        on either side of the cut-off).
      * Double click outside axes saves the truncated time trace to file."""

    def __init__(self, pathfname, time, channels, labels, relative_margin=0.1):
        # initialise attributes:
        self.pathfname, self.margin = pathfname, relative_margin
        self.time, self.chs, self.labels = time, channels, labels
        self.lower_ix, self.upper_ix = 0, -1
        # calculate scaled power spetral densities
        self.truncate()
        # initialise the figure and connect the button_press_event
        self.fig, self.ax = self.initialise_plot()
        self.canvas = self.fig.canvas
        self.cid = self.canvas.mpl_connect('button_press_event', self.onclick)
        # update to plot the data and show the figure
        self.update_plot()
        plt.show()

    def truncate(self):
        """Calculate the truncated timeseries and spectra"""
        self.trunc_time = self.time[self.lower_ix:self.upper_ix]
        self.trunc_chs = [ch[self.lower_ix:self.upper_ix] for ch in self.chs]

    def initialise_plot(self, figsize=(8, 9)):
        """Create figure and axis objects, plot data and return objects"""
        fig = plt.figure(figsize=figsize)
        # creating subplots of different size:
        gs = gridspec.GridSpec(3, 2)
        ax = [plt.subplot(gs[0, :]),
              plt.subplot(gs[2, :]),
              plt.subplot(gs[1, 0]),
              plt.subplot(gs[1, 1])]
        self.add_labels(ax)
        plt.tight_layout()
        return fig, ax

    def onclick(self, event):
        """Handling a click event: save truncated series if double click,
        otherwise set new limits"""
        if event.dblclick: # save the current truncated version
            print("\nSave the current truncated time series trace with to")
            print("%s_[postfix]%s" % (self.pathfname, _filetype))
            postfix = input("Specify [postfix], or n+enter to abort: ")
            if postfix.lower() == 'n':
                print("Not saved")
            else:
                date_time = str(datetime.now()) # get current date and time
                head = "time,"+','.join(self.labels)+"\n"
                # shape data in correct from
                x = np.array([self.trunc_time]).T
                y = np.array([ch for ch in self.trunc_chs]).T
                data = np.append(x, y, axis=1)
                np.savetxt(self.pathfname+"_"+postfix+_filetype, data, delimiter=",", header=head+date_time)
                plt.savefig(self.pathfname+"_"+postfix)
                print("Saved")
        elif event.inaxes in self.ax: # click in any of the axes
            ix, _ = find_nearest(self.time, event.xdata)
            if event.inaxes == self.ax[0]: # click in original timeseries
                if event.button == 1: # if left click
                    self.set_ix(new_l=ix)
                elif event.button == 3: # right click
                    self.set_ix(new_u=ix)
            elif event.inaxes == self.ax[1]: # click in lower lim plot
                self.set_ix(new_l=ix)
            elif event.inaxes == self.ax[3]: # click in upper lim plot
                self.set_ix(new_u=ix)
            # redraw the plots with updated indecies
            self.update_plot()

    def set_ix(self, new_l=None, new_u=None):
        """Set new lower/upper indices if new indecies span a range"""
        if new_l is not None and (new_l < self.upper_ix or self.upper_ix == -1): self.lower_ix = new_l
        if new_u is not None and new_u > self.lower_ix: self.upper_ix = new_u

    def update_plot(self, alpha=0.8):
        """Redraw the axes with
            * lines indicating the points selected so far
            * updated truncated timeseries
            * updated spectra"""
        # calculate truncated spectra
        self.truncate()
        for a in self.ax: a.clear()
        # plot the time series and spectra
        for ch, label in zip(self.chs, self.labels):
            self.ax[0].plot(self.time, ch, label=label, alpha=alpha)
            self.ax[2].plot(self.time, ch-np.mean(ch), label=label, alpha=alpha)
            self.ax[3].plot(self.time, ch-np.mean(ch), label=label, alpha=alpha)
        # plot truncated time series
        for ch in self.trunc_chs:
            self.ax[1].plot(self.trunc_time, ch, alpha=alpha)
        # plot axislines for the cut offs and enable gridlines on spectrum
        for a in [self.ax[0], self.ax[2], self.ax[3]]:
            a.axvline(self.time[self.lower_ix], c='r')
            a.axvline(self.time[self.upper_ix], c='b')
        # set limits for smaller plots:
        time_margin = self.margin*(time[-1]-time[0])
        for a, ix in zip([self.ax[2], self.ax[3]], [self.lower_ix, self.upper_ix]):
            a.set_xlim(self.time[ix]-time_margin, self.time[ix]+time_margin)
        # add labels and draw the canvas
        self.add_labels()
        self.canvas.draw()

    def add_labels(self, ax=None, freq_lims=[0, 200], spectrum_lims=[1e-12, 1e-5]):
        """Set the axis labels and limits of axes in the vector ax, or
        when None uses self.ax"""
        ax = ax if ax is not None else self.ax
        ax[0].legend()
        for a in ax[0:2]:
            a.set_xlabel("Time [s]")
            a.set_ylabel("Transmission signal")
        ax[0].set_title("Original time series")
        ax[0].set_xlim([min(self.time), max(self.time)])
        ax[1].set_title("Truncated time series")
        ax[1].set_xlim([min(self.trunc_time), max(self.trunc_time)])
        ax[2].set_title("Lower cut-off")
        ax[3].set_title("Upper cut-off")

def find_nearest(array, value):
    """Return index and value closest to 'value' in 'array'"""
    array = np.asarray(array)
    ix = (np.abs(array - value)).argmin()
    return ix, array[ix]


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
##                                   MAIN                                     ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##


folder = "./data/"
fnames = ["20190912(1) 1550.0+52.3V_opposite_trig_20Hz_mment57"]
# fnames = ["15"+str(i)+" AVER4" for i in range(53,56)]

if __name__ == '__main__':
    for fname in fnames:
        time, ch1, ch2 = read_data(folder+fname, usecols=[0,1,2], names=['time', 'ch1', 'ch2'])
        chs = (ch1, ch2)
        labels = ("ch1", "ch2")
        SetTimeseriesWindowing(folder+fname, time, chs, labels)
