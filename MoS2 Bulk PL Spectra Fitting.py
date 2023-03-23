#Import Modules 
import numpy as np #pip install numpy
from numpy import loadtxt
import matplotlib.pyplot as plt #pip install matplotlib
import csv
import scipy #pip install scipy
import glob
import os

#define Lorentzian function: x = x_data, int = peak intensity / a.u, x_0 = peak position / eV, w = (fwhm / 2) / eV
def lorentzian(x, int, x_0, w):
    return (int*(w**2)/((x-x_0)**2+(w**2)))

#define two Lorentzian function model: x = x_data, int = peak intensity / a.u, x_0 = peak position / eV, w = (fwhm / 2) / eV for A and B emission peaks
def _2lorentzian(x, int_A, x_0_A, w_A, int_B, x_0_B, w_B):
    return ((int_A*(w_A**2)/((x-x_0_A)**2+(w_A**2))) + (int_B*(w_B**2)/((x-x_0_B)**2+(w_B**2))))

results_list=[] #define list to store results

#Set starting expected peak params of form: [Peak Intensity / a.u, peak position / eV, (FWHM/2) /eV]
A_peak = [9000, 1.85, 0.1]
B_peak = [3200, 2, 0.15]

#loop over all spectra .txt files in defined folder
for filename in glob.glob('PL MoS2 Data/*.txt'):

    #trim filename from filepath
    head, tail = os.path.split(filename)
    spectra_name = tail[:-4]
    spectra_name_result = np.array([spectra_name]) #append name to 1D array to output with results

    #load spectra data into np array ignoring headers
    spectra_data = loadtxt(filename)

    #extract all x data range
    x = spectra_data[0:len(spectra_data)+1,0] 
    #extract all y data range
    y = spectra_data[0:len(spectra_data)+1,1]

    #optimise curve fit to spectrum using 2 lorentzian function and A and B initial params
    popt, pcov = scipy.optimize.curve_fit(_2lorentzian, x, y, p0 = [A_peak, B_peak])
    perr = np.sqrt(np.diag(pcov))

    #get residual sum of squares and calculate r^2
    residuals = y - _2lorentzian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - (ss_res/ss_tot)

    #save r2 value of _2lorentzian fit to 1D array to be output with results
    r2_result = np.array([r2])

    #unpack optimised parameters and errors from A and B peaks
    A_params = popt[0:3]
    A_errors = perr[0:3]
    B_params = popt[3:6]
    B_errors = perr[3:6]

    #unpack intensity and position, and errors for A.
    A_params_copy = A_params.copy() #make copy so we can correct w to fwhm
    A_errors_copy = A_errors.copy() #make copy so we can correct error of w to error of fwhm

    A_int_x0 = A_params_copy[0:2] #get optimised intensity and peak position values for A peak
    A_int_x0_errors = A_errors_copy[0:2] #get optimised intensity and peak position errors for A peak
    
    A_fwhm = np.array([2*A_params_copy[2]]) #correct w to fwhm for A peak
    A_fwhm_error = np.array([2*A_errors_copy[2]]) #correct w error to fwhm error for A peak

    #concatenate all results for current spectra into 1D array (1 row containing all results)
    results = np.concatenate((spectra_name_result, A_int_x0, A_int_x0_errors, A_fwhm, A_fwhm_error, r2_result)) 
    results_list.append(results) #append results from spectra to list (1 row per spectra, cumulative)

    #plot and save spectra
    plt.plot(x, y, 'black') #plot raw spectrum data
    plt.plot(x, lorentzian(x, *A_params), 'g--', label = 'A Peak Lorentzian') #plot fit to A peak
    plt.plot(x, lorentzian(x, *B_params), 'g--', label = 'B Peak Lorentzian') #plot fit to B peak
    plt.plot(x, _2lorentzian(x, *popt), 'r--', label = 'Lorentzian fit') #plot combined fit to spectrum
    plt.xlabel('Energy / eV', family = 'serif', fontsize =12) #add x axis label
    plt.ylabel('PL Intensity (a.u.)', family = 'serif', fontsize =12) #add y axis label
    plt.legend(loc="best") #add legend
    plt.savefig(f"{spectra_name} fitted spectra.png") # save the figure to file
    plt.close()

#--------------EXPORT DG FITTING RESULTS TO .CSV FILE-------------------#
#Create tab delimiter .csv file tabulating flake morphological parameter data
headers = ["Spectra Name", "A Peak Intensity / a.u", "A peak position / eV", "A Peak Intensity error / a.u", "A peak position error / eV", "A peak fwhm / eV", "A peak FWHM error /eV", "R^2"] #define collumn headers
with open('MoS2 PL Fitted A Peaks.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(results_list)
