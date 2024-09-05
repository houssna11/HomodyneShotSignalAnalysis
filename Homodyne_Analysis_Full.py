# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:30:49 2024

@author: houss
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simps
from scipy.optimize import curve_fit
from multiprocessing import Pool
from datetime import datetime
import os

# Replace with your actual file path
file_path ="F:\Data_25062024\Homodyne_Vaccum_12_20240625_164510.csv"
chunksize = 10000000000000000000000  # Adjust based on available memory

# Read the first chunk to plot the first millisecond of 'Input 1'
first_chunk = pd.read_csv(file_path, skiprows=18, names=['Time (s)', 'Input 1 (V)', 'Input 2 (V)'], nrows=chunksize)
time_first_ms = first_chunk[first_chunk['Time (s)'] <= 0.005]['Time (s)']
input1_first_ms = first_chunk[first_chunk['Time (s)'] <= 0.005]['Input 1 (V)']

plt.figure(figsize=(10, 5))
plt.plot(time_first_ms, input1_first_ms)
plt.xlabel('Time (s)')
plt.ylabel('Input 1 (V)')
plt.title('First Millisecond of Input 1 (V)')
plt.grid(True)
plt.show()


#%%
# Read the first chunk to plot the first millisecond of 'Input '
first_chunk = pd.read_csv(file_path, skiprows=18, names=['Time (s)', 'Input 1 (V)', 'Input 2 (V)'], nrows=chunksize)
time_first_ms = first_chunk[first_chunk['Time (s)'] <= 0.005]['Time (s)']
input2_first_ms = first_chunk[first_chunk['Time (s)'] <= 0.005]['Input 2 (V)']

plt.figure(figsize=(10, 5))
plt.plot(time_first_ms, input2_first_ms)
plt.xlabel('Time (s)')
plt.ylabel('Input 2 (V)')
plt.title('First Millisecond of Input 2 (V)')
plt.grid(True)
plt.show()


#%%
# Parameters
threshold_positive = 0.000# Positive peak threshold for Input 1
threshold_negative = -0.1# Negative peak threshold for Input 2
integration_window = 0.00007  # Integration window duration in seconds
integration_delay = 0.000002  # Delay after peak detection before integration starts
sampling_rate = 1e6  # 1 MHz
delay_points = int(integration_delay * sampling_rate)
integration_points = int(integration_window * sampling_rate)
chunksize = 10000000000000000000000  # Adjust based on available memory

# Read the CSV file in chunks
data_chunks = pd.read_csv(file_path, skiprows=18, names=['Time (s)', 'Input 1 (V)', 'Input 2 (V)'], chunksize=chunksize)

# Initialize variables to store results
integrals = []
peak_amplitudes = []
all_peaks = []
groups = {}

# Gaussian fitting function
def gaussian(x, amp, mean, sigma):
    return amp * np.exp(-((x - mean)**2 / (2 * sigma**2)))

# Function to print sigma
def print_sigma(parameters):
    amp, mean, sigma = parameters
    print(f"The calculated sigma (standard deviation) for the non-filtered Gaussian is: {sigma:.10f}")


# Process each chunk
for chunk_index, chunk in enumerate(data_chunks):
    # Detect positive peaks in Input 1
    positive_peaks, props = find_peaks(chunk['Input 1 (V)'], height=threshold_positive)
    # Adjust peak indices to match the global data indices
    global_peaks = positive_peaks + chunk_index * chunksize
    all_peaks.extend(global_peaks)
    
    # Integrate the signal for each detected peak
    for peak in positive_peaks:
        start_index = peak + delay_points
        end_index = start_index + integration_points
        if end_index < len(chunk):
            integral = simps(chunk['Input 1 (V)'].iloc[start_index:end_index], chunk['Time (s)'].iloc[start_index:end_index])
            integrals.append((integral, chunk['Input 1 (V)'].iloc[peak]))
# Generate histogram and fit Gaussian
plt.figure(figsize=(10, 5))
hist, bin_edges = np.histogram([integral for integral, _ in integrals], bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
p0_amp = hist.max()
p0_mean = np.average(bin_centers, weights=hist)
p0_sigma = np.sqrt(np.average((bin_centers - p0_mean)**2, weights=hist))
popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[p0_amp, p0_mean, p0_sigma])

# Histogram of integrals
plt.hist([integral for integral, _ in integrals], bins=50, edgecolor='black', alpha=0.5, density=True)

# Gaussian fit
x_fit = np.linspace(bin_edges[0], bin_edges[-1], 1000)
y_fit = gaussian(x_fit, *popt)
plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit', linewidth=2)

# Highlight area under the Gaussian fit
plt.fill_between(x_fit, 0, y_fit, color='blue', alpha=0.3, label='Fit Area')

# Styling
plt.title('Gaussian Fit for All Integrals')
plt.xlabel('Integral Value')
plt.ylabel('Density')
plt.legend(frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

print_sigma(popt)

if all_peaks:
    first_peak = all_peaks[0]
    start_index = first_peak + delay_points
    end_index = start_index + integration_points
    chunk_index = first_peak // chunksize
    local_peak = first_peak % chunksize
    chunk = pd.read_csv(file_path, skiprows=18 + chunk_index * chunksize, nrows=chunksize, names=['Time (s)', 'Input 1 (V)', 'Input 2 (V)'])

    plt.figure(figsize=(10, 5))
    plt.plot(chunk['Time (s)'], chunk['Input 1 (V)'], label='Signal', color='black')
    plt.axvline(chunk['Time (s)'].iloc[local_peak], color='g', linestyle='--', label='Peak')
    plt.axvline(chunk['Time (s)'].iloc[start_index], color='r', linestyle='--', label='Start of Integration')
    plt.axvline(chunk['Time (s)'].iloc[end_index], color='b', linestyle='--', label='End of Integration')
    plt.fill_betweenx(chunk['Input 1 (V)'], chunk['Time (s)'].iloc[start_index], chunk['Time (s)'].iloc[end_index], color='r', alpha=0.3, label='Integration Area')
    plt.xlim(chunk['Time (s)'].iloc[local_peak] - 50e-6, chunk['Time (s)'].iloc[end_index] + 50e-6)
    plt.xlabel('Time (s)')
    plt.ylabel('Input 1 (V)')
    plt.title('First Detected Peak with Integration Region')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
#%%
chunksize = 10000000000000000000000  # Adjust based on available memory
responsivity = 0.52  # A/W for 666 nm
transimpedance_gain = 1e6  # V/A
repetition_rate = 1000

# Function to process each chunk and calculate energy
def process_chunk(chunk):
    # Filter peaks below -0.005 V (considering absolute value to identify peaks)
    peaks = chunk[chunk['Input 2 (V)'] < -0.04].copy()
    # Integrate the area under the peaks to find the energy
    if not peaks.empty:
        peaks['Energy (V*s)'] = np.abs(np.trapz(peaks['Input 2 (V)'], peaks['Time (s)']))
    return peaks

# Initialize an empty DataFrame to store results
all_peaks = pd.DataFrame()

# Read the CSV in chunks
for chunk in pd.read_csv(file_path, skiprows=18, names=['Time (s)', 'Input 1 (V)', 'Input 2 (V)'], chunksize=chunksize):
    chunk_peaks = process_chunk(chunk)
    all_peaks = pd.concat([all_peaks, chunk_peaks])

# Convert integrated voltage to current using transimpedance gain
all_peaks['Energy (A*s)'] = all_peaks['Energy (V*s)'] / transimpedance_gain
# Convert current to energy using responsivity (Joules per pulse)
all_peaks['Energy (J)'] = all_peaks['Energy (A*s)'] / responsivity

# Average energy per pulse
average_energy_per_pulse = all_peaks['Energy (J)'].mean()

# Considering the laser repetition rate to get average power
average_power_watts = average_energy_per_pulse * repetition_rate

print(f"Average Power of the Registered Peaks: {average_power_watts:.10f} W")
