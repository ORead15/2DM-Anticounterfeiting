import numpy as np #pip install numpy
from numpy import loadtxt
import rampy as rp #pip install rampy
import matplotlib.pyplot as plt #pip install matplotlib
import csv
import scipy #pip install scipy
import glob
import os

#define Lorentzian function: x = x_data, int = peak intensity / a.u, x_0 = peak position / cm^-1, w = (fwhm / 2) / cm^-1
def lorentzian(x, int, x_0, fwhm):
    return (int*(fwhm**2)/((x-x_0)**2+(fwhm**2)))

#define two Lorentzian function model: x = x_data, int = peak intensity / a.u, x_0 = peak position / cm^-1, w = (fwhm / 2) / cm^-1 for E12g and A1g peaks
def _2lorentzian(x, int_e2g, x_0_e2g, fwhm_e2g, int_a1g, x_0_a1g, fwhm_a1g):
    return ((int_e2g*(fwhm_e2g**2)/((x-x_0_e2g)**2+(fwhm_e2g**2))) + (int_a1g*(fwhm_a1g**2)/((x-x_0_a1g)**2+(fwhm_a1g**2))))

results_list = [] #define list to store results

#Set starting expected peak params of form: [intensity / a.u, peak position / cm^-1, (fwhm/2) cm^-1]
e12g_peak = [1500, 385, 5]
a1g_peak = [3000, 404, 6]

#loop over all spectra .txt files in defined folder
for filename in glob.glob('Raman MoS2 Data/*.txt'):
    
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

    #remove background and zero baseline using arPLS algorithm
    bir = np.array([(352,375),(420,452)])  #set baseline regions
    y_cor, background = rp.baseline(x,y, bir,"arPLS",lam=10**9) #lam needs to be optimised, typically between 10**4-10**9
    y = y_cor[:,0] #set corrected y data to y

    #optimise curve fit to spectrum using 2 lorentzian function and e12g and a1g initial params
    popt, pcov = scipy.optimize.curve_fit(_2lorentzian, x, y, p0 = [e12g_peak, a1g_peak])
    perr = np.sqrt(np.diag(pcov))

    #get residual sum of squares and calculate r^2
    residuals = y - _2lorentzian(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - (ss_res/ss_tot)

    #save r2 value of _3lorentzian fit to 1D array to be output with results
    r2_result = np.array([r2])

    #unpack optimised parameters for e12g and a1g peaks
    e12g_params = popt[0:3]
    a1g_params = popt[3:6]

    #calculate peak to peak difference by subtracting a1g peak position from e12g position
    peak_diff = a1g_params[1] - e12g_params[1]

    peak_diff_result = np.array([peak_diff]) #add peak diff to numpy array to output with results

    #estimate layer number using peak to peak diff
    if peak_diff == 0:
        n = 0
    if peak_diff < 20:
        n = 1
    if peak_diff > 20 < 22:
        n = 2
    if peak_diff > 22 < 23:
        n = 3
    if peak_diff > 23:
        n = 4

    layernumber_result = np.array([n]) #add layernumber result to 1d numpy array to output with results

    #concatenate all results for current spectra into 1D array (1 row containing all results)
    results = np.concatenate((spectra_name_result, peak_diff_result, layernumber_result, r2_result)) 
    results_list.append(results) #append results from spectra to list (1 row per spectra, cumulative)

    #plot and save spectra
    plt.plot(x, y, 'black') #plot raw spectrum data
    plt.plot(x, lorentzian(x, *e12g_params), 'g--', label = "E12g Lorentzian fit") #plot fit to e12g peak
    plt.plot(x, lorentzian(x, *a1g_params), 'g--', label = "A1g Lorentzian fit") #plot fit to a1g peak
    plt.plot(x, _2lorentzian(x, *popt), 'r--', label = 'Lorentzian fit') #plot combined fit to spectrum
    plt.xlabel('Raman Shift / cm^-1', family = 'serif', fontsize =12) #add x axis label
    plt.ylabel('Counts (a.u.)', family = 'serif', fontsize =12) #add y axis label
    plt.legend(loc="best") #add legend
    plt.savefig(f"{spectra_name} fitted spectra.png") # save the figure to file
    plt.close()

#--------------EXPORT DG FITTING RESULTS TO .CSV FILE-------------------#
#Create tab delimiter .csv file tabulating flake morphological parameter data
headers = ["Spectra Name", "Peak Difference / cm^-1", "Layer number", "R^2"] #define collumn headers
with open('MoS2 Raman Fitted Peaks.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(results_list)
