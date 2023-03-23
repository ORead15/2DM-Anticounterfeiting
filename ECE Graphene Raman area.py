#import all modules
import numpy as np #pip install numpy
from numpy import loadtxt
import matplotlib.pyplot as plt #pip install matplotlib
import scipy as scipy #pip install scipy
import rampy as rp #pip install rampy
import os

#cosmic ray removal function adapted from article by Nicolas Coca: https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
#function to calculate modified z-score for cosmic ray removal
def modified_z_score(intensity):
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores

#function to remove cosmic rays from spectra 
def remove_cosmic_rays(y, m):
    #peaks above threshold are cosmic ray spikes (set to 1), peaks below are not(set to 0)
    threshold = 11 #binaraization threshold
    spikes = abs(np.array(modified_z_score(np.diff(y)))) > threshold
    y_out = y.copy() #copy y data to prevent overwriting

    for i in np.arange(len(spikes)):
        if spikes[i] != 0: #if we have a spike (=1) in position i
            w = np.arange(i-m, i+1+m) #we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From 2m+1 points, we chose the indexes which do not correspond to spikes
            y_out[i] = np.mean(y[w2]) #we average the y values corresponding to non-spike indexes.
            
    return y_out

#---------------LOAD MAPPING DATA, INITIALISE MAPPING MATRIX PARAMETERS, DEFINE RESULTS MATRICES--------------------------------#

#Load mapping dataset, with raw data of form: x coor, y coor, raman shift / cm^-1, intensity / a.u.
filename = 'path_to_raw_mapping_data.txt' #define mapping data filename

#load spectra data into np array ignoring headers
spectra_data = loadtxt(filename)

#trim sample name and output folder from filepath and remove .txt extension
head, tail = os.path.split(filename)
sample_name = tail[:-4]
output_folder = head

#find matrix dimensions of raman map
matrix_x_dimension = np.size(np.unique(spectra_data[:,0]))   #find length of the array of containing all unique x coordinate values 
matrix_y_dimension = np.size(np.unique(spectra_data[:,1]))   #find length of the array of containing all unique y coordinate values

#calculate number of spectra in the map i.e number of pixels
spectra_count = matrix_x_dimension * matrix_y_dimension

#define number of rows per spectra or spectral range i.e number of unique data points per spectra
rows = 1011

#define empty matrices of equal size to raman map to store results in
area_matrix = np.zeros((matrix_x_dimension, matrix_y_dimension))
threshold_matrix = np.zeros((matrix_x_dimension, matrix_y_dimension))

#---------------------MAIN SCRIPT LOOP------------------------------------------------------------#
n = matrix_y_dimension  #rename parameter for modifier loops
j_x = 0     #initialise modifier for x-coord
j_y = 0     #initialise modifier for y-coord
x_start = min(spectra_data[:,0]) #initialise x_data_range starting data point
y_start = max(spectra_data[:,1]) #initalise y_data_range starting data point
spectra_fitted = 0 #start count

#Loop through each spectra of the map 
for i in range(0,spectra_count):
    print(f"i is: {i} ")
    #calculate start and end row indexes for spectra i (e.g start at 0, end at 1011, start at 1012 end at 2023 etc)
    start = i*rows
    end = (i+1)*rows
    
    #get x and y coordinate i.e pixel locations of that spectra, then modify to convert its raman coordinates to matrix coordinates
    
    #if remainder of i / n = 0, add 1 to j_x  (i.e. every n loops change the j_x modifier)
    if i % n == 0:          
        j_x += 1
    if i == 0:     #error catch, ensure 0 / n doesn't mess up the modifier count.
        j_x = 0
    
    #read raman matrix x position, adjust to start at 0, then add modifier to make integer value.
    x_coord = round((spectra_data[start, 0]) + abs(x_start) + j_x*0.4) #0.4 parameters may need to be changed depending on Raman measurement settings.
    print(f"x_coord is: {x_coord}")
    #every n loops reset j_y modifier i.e move down a row after completing the point in each column
    if i % n == 0:
        j_y = 0    
    
    #read raman matrix y position, adjust to start at 0, then add modifier to make integer value
    y_coord = round(((spectra_data[start, 1]) - y_start + (j_y*1.6))) #1.6 parameter may need to be changed depending on Raman measurement settings.
    j_y += 1 #add one to count
    print(f"y_coord is: {y_coord}")
    
    #crop x data (raman shift / cm^-1) and y data (counts / a.u.) for spectra i from raw mapping data
    #using calculated start and end points
    x = spectra_data[start:end, 2]
    y = spectra_data[start:end, 3]

    #remove cosmic rays from y data
    y = remove_cosmic_rays(y, m = 3) #y is intensity data, m is variable used to remove cosmic ray and replace with average of 2m+1 neighbouring data points

    #smooth spectra (if required), lambda must be optimised ensuring no data #e.g peak intensity is lost from excess smoothing. 
    #lambda = 0 (no smoothing) up to lambda = 1000.
    y = rp.smooth(x,y,method="whittaker",Lambda=1) #use Whittaker algorithm to apply minimum smoothing

    #remove background automatically using arPLS algorithm
    bir = np.array([(1079,1200),(1700,2275)])  #set baseline regions
    y_cor, background = rp.baseline(x,y, bir,"arPLS",lam=10**8) #lam needs to be optimised, typically between 10**4-10**9
    y = y_cor[:,0] #set corrected y data to y

    area = np.trapz(y) #get area under curve by integrating using trapezium rule
    area_matrix[x_coord, y_coord] = area #set area value to corresponding position in results matrix
    
    #to produce threshold Raman map
    if area > 15000: #set binarisation threshold, may need to be optimised for different data
        threshold_matrix[x_coord, y_coord] = 1 #set pixels/spectra associated with nanosheet to 1
    else:
        threshold_matrix[x_coord, y_coord] = 0 #set pixels/spectra associated with background to 0

#rotate results matrices to match corresponding optical images  
threshold_matrix = np.rot90(threshold_matrix, k = 1, axes = (0,1))
area_matrix = np.rot90(area_matrix, k = 1, axes = (0,1))

#save mapping images
plt.imshow(threshold_matrix, cmap = 'bone')#load threshold map
plt.axis('off') #turn off axis
plt.clim(0,1) #set scale
plt.savefig(f"{output_folder}/{sample_name} threshold map.tif", bbox_inches ='tight', pad_inches = 0)#save image with no scale or whitespace to same folder containing loaded spectra
plt.close() #close figure

plt.imshow(area_matrix, cmap = 'bone')#load raman area map
plt.axis('off') #turn off axis
cb = plt.colorbar(label = 'A(D+G+D\')') # add colorbar with label
plt.tight_layout() #use a tight layout
plt.savefig(f"{output_folder}/{sample_name} Raman area.tif", bbox_inches ='tight', pad_inches = 0) #save final DG ratio map into the same folder containing the loaded spectra, using the filename as the name.
plt.close()
