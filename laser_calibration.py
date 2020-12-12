# -*- coding: utf-8 -*-
"""
Using the qfactor module to extract the laser calibration for internal scanning
of either the current or the voltage.

For external scanning, change the scaling parameter according to what
is set for the lasers' input.

Andreas Svela // Dec 2020
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt

import qfactor


def laser_calibration_finder(**kwargs):
    qfm = qfactor.QFactorMeasurement(**kwargs)
    scan_freqs = []
    scan_amplitudes = []
    calibrations = []
    while True:
        scan_amplitude = float(input("Scan peak to peak amplitude? [V or mA] "))
        scan_freq = float(input("Scan frequency? [Hz] "))
        calibration = qfactor.calc_calibration(qfm.frequency_span_per_sec, scan_freq,
                                               scan_amplitude, scaling=1)
        scan_amplitudes.append(scan_amplitude)
        scan_freqs.append(scan_freq)
        calibrations.append(calibration)
        print(f"Calibration: {calibration:,.2f} MHz/V or MHz/mA")
        cont = input("Do you wish to make another measurement? (Y/n): ")
        if cont.lower() == 'n':
            break
        try:
            qfm.set_frequency_span_per_sec()
        except RuntimeError as e:
            print(e)
            break
    print(f"freqs: {scan_freqs}")
    print(f"amps:  {scan_amplitudes}")
    print(f"cals:  {calibrations}")
    df = pd.DataFrame(np.array([scan_freqs, scan_amplitudes, calibrations]).T, columns=["scan freq", "scan amplitude", "calibration"])
    save(df)
    plots(df)

def save(df):
    timestamp = dt.datetime.now().strftime("%y%m%d_%Hh%Mm%S")
    df.to_csv(f"{timestamp} laser calibration.csv")

def plots(df, scan_unit="V", common_scatter_plot=False):
    fscan_label = "Scan frequency [Hz]"
    ascan_label = f"Scan amplitude [{scan_unit}]"
    calib_label = f"Calibration [MHz/{scan_unit}]"
    df = df.drop_duplicates(subset=["scan freq", "scan amplitude"], keep="last")
    # pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')
    ## Separate scatter plots
    fig, ax = plt.subplots(ncols=2, figsize=(13, 5))
    df.plot.scatter(ax=ax[0], x="scan amplitude", y="calibration", c="scan freq", colormap="cividis", edgecolor="k", lw=0.2)
    df.plot.scatter(ax=ax[1], x="scan freq", y="calibration", c="scan amplitude", colormap="magma", edgecolor="k", lw=0.2)
    ax[0].set_xlabel(ascan_label)
    ax[0].set_ylabel(calib_label)
    ax[1].set_xlabel(fscan_label)
    ax[1].set_ylabel(calib_label)
    # Linear fit for calibration with scan amplitude
    def linear(x, x0, x1):
        return x0 + x*x1
    popt, _ = opt.curve_fit(linear, df["scan amplitude"].values, df["calibration"].values)
    amp_range = np.linspace(df["scan amplitude"].min(), df["scan amplitude"].max())
    fitline = linear(amp_range, *popt)
    ax[0].plot(amp_range, fitline, 'k--', label=f"{popt[0]:.0f} + x*{popt[1]:.3f}")
    ax[0].legend()
    if common_scatter_plot:
        fig, ax = plt.subplots()
        sc = ax.scatter(df["scan freq"], df["scan amplitude"], c=df["calibration"])
        ax.set_xlabel(fscan_label)
        ax.set_ylabel(ascan_label)
        cbar = fig.colorbar(sc)
        cbar.set_label(calib_label)
    ## Interpolation plot
    # First drop duplicates of (x, y) pairs as Rbf can't handle that
    df = df.drop_duplicates(subset=["scan freq", "scan amplitude"], keep="last")
    # Shorthands
    freq, amp, cal = df["scan freq"].values, df["scan amplitude"].values, df["calibration"].values
    # Define the parameter space
    extent = [np.linspace(df[col].min()*0.90, df[col].max()*1.02, 500) for col in ("scan freq", "scan amplitude")]
    ff, aa = np.meshgrid(*extent)
    # Interpolate
    rbf = interp.Rbf(freq, amp, cal, function='linear')
    cc = rbf(ff, aa)
    # Plot
    fig, ax = plt.subplots()
    im = ax.pcolormesh(ff, aa, cc, shading="auto")
    ax.set_xlabel(fscan_label)
    ax.set_ylabel(ascan_label)
    cbar = fig.colorbar(im)
    cbar.set_label(calib_label)
    clims = im.get_clim()
    sc = ax.scatter(freq, amp, c=cal, edgecolor='w', lw=1, vmin=clims[0], vmax=clims[1])
    plt.show()

def apollo_voltage_data():
    scan_freqs = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 100.0, 75.0, 25.0, 10.0, 10.0, 10.0, 10.0, 10.0,  25.0,   50.0,  75.0, 100.0, 100.0, 75.0, 50.0, 25.0,  5.0, 25.0, 50.0, 10.0, 10.0,  10.0,  25.0,  50.0,  75.0, 100.0, 100.0,  75.0,  50.0,  25.0,  10.0, 25.0, 50.0, 75.0, 100.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 87.0, 87.0, 87.0, 87.0, 87.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 75.0, 87.0, 100.0, 100.0, 87.0, 87.0, 100.0, 75.0, 75.0, 87.0, 100.0]
    scan_amplitudes = [2.0, 5.0, 10.0, 20.0, 50.0, 50.0, 25.0, 10.0,  5.0,  2.5,  1.0,   2.5,  2.5,  2.5,  2.5, 10.0, 25.0, 50.0, 100.0, 100.0, 100.0, 100.0, 100.0,  50.0, 50.0, 50.0, 50.0, 50.0, 25.0, 25.0, 120.0, 75.0, 140.0, 140.0, 140.0, 140.0, 140.0, 120.0, 120.0, 120.0, 120.0, 120.0, 75.0, 75.0, 75.0,  75.0, 25.0,  25.0,  25.0, 50.0, 75.0, 75.0, 100.0, 120.0, 120.0, 100.0, 75.0, 50.0, 25.0, 25.0, 50.0, 50.0, 75.0, 100.0, 120.0, 140.0, 50.0, 50.0, 50.0, 75.0, 75.0, 62.0, 62.0, 62.0, 37.0, 37.0, 37.0]
    calibrations = [174.14679332802183, 178.5391592513075, 187.1384054332406, 200.4997559427305, 238.88314258084597, 230.86953714169877, 206.07202206129847, 190.48735410376443, 182.4859681773067, 179.20929169861478, 173.1267570473021, 176.66741924331282, 176.19537158363923, 178.5850532303106, 178.363063874018, 190.33908607818464, 210.7575822629209, 235.94934402007172, 267.44710163213733, 264.40207654012454, 272.8748792342344, 241.84292340201603, 200.7885781815517, 186.85633353276077, 222.38648357048737, 230.86953714169877, 235.91935862872168, 238.8444773284724, 212.05954979715906, 206.07202206129844, 277.14, 254.83, 293.72, 291.23, 282.36, 281.83, 283.96, 235.92, 246.86, 272.56, 271.83, 272.14, 251.7797240220374, 252.85773257630703, 255.31330416299932, 245.20849803034483, 207.23705472218663, 215.76387461005973, 219.59567546886953, 274.67951373027523, 210.06233416992575, 210.06233416992575, 285.34642814487944, 274.6620933028429, 285.61218064364704, 278.3834834788962, 260.4063003960809, 269.91417361838614, 228.55098812450507, 193.1497513043335, 7549.4834175331725, 214.0553856210992, 226.58555807213614, 285.0742997949896, 284.4734433090074, 296.5440295922924, 214.0553856210992, 269.91417361838614, 274.67951373027523, 210.06233416992575, 289.99017001396123, 276.55488074248797, 245.0803946747757, 253.37332309118162, 231.89776036078797, 256.44337732240183, 217.37224393584137]
    df = pd.DataFrame(np.array([scan_freqs, scan_amplitudes, calibrations]).T, columns=['scan freq', 'scan amplitude', 'calibration'])
    save(df)
    plots(df)

if __name__ == "__main__":
    # apollo_voltage_data()
    laser_calibration_finder(PD_zero="vacuum", acq_type="AVER32")
