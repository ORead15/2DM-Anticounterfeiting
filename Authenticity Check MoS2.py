#import modules
import cv2 #pip install opencv-ython
import os
import mahotas #pip install mahotas
import numpy as np #pip install numpy
import glob
from scipy.spatial import distance as dist #pip install scipy
from matplotlib import pyplot as plt #pip install matplotlib
import csv


#define function to return contour and Zernike moments from an image of a nanosheet
#Note MoS2 image dataset requires erosion and dilation of 1 pixel from external contour to prevent merging of neighbouring features
def get_moments(filename):
    zernikeMoments = [] #initialise list to store zernike moments
  
    image = cv2.imread(filename) #read image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.5) #blur
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #convert to binary
    
    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]
    
    #create an empty mask for the contour and draw filled contour
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)
  
    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]

    kernel = np.ones((3,3), np.uint8) #set kernel

    mask = cv2.erode(mask, kernel) #erode 1 pixel from external contour

    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]

    #create an empty mask for the contour and draw filled contour
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)

    mask = cv2.dilate(mask, kernel) #dilate 1 pixel to external contour

    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]

    #compute Zernike Moments from roi, add to list.
    zMoments = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(cnts)[1], degree=8)
    zernikeMoments.append(zMoments)
       
    #return contour and its Zernike moments
    return (cnts, zernikeMoments)

#define function to return L, W, A and L/W from a nanosheet contour
def measure_size(flake_contour):
    #calculate pixel size and area
    pixel_size = 1.0 / image_scale # in um
    pixel_area = pixel_size**2 #in um^2
    
    #calculate flake area in pixels and convert to um^2.
    flake_area = cv2.contourArea(flake_contour) #counts all pixels in contour
    flake_area = flake_area * pixel_area #unit conversion to um^2

    #Calculate flake L, W and L/W
    bounding_rect = cv2.minAreaRect(flake_contour) #create bounding rectangle with minimum area around contour
    (x,y), (w,h), angle = bounding_rect #x,y centre points of rectangle, w h, width and height of rectangle in pixels
    #take largest rectangle dimension as L and the shorter as W
    if h > w:
        lateral_size = h
        width = w
    else: 
        lateral_size = w 
        width = h
    lateral_size = lateral_size * pixel_size #convert units from pixels to um
    width = width * pixel_size #converts units from pixels to um
    
    ratio = lateral_size / width #compute aspect ratio

    return flake_area, lateral_size, width, ratio

image_scale = 12.9 #set image scale in pixels/um (measure from scale bar)

#Define Reference Image i.e. nanosheet to check for authenticity
ref_filename = "MoS2 Nanotag Duplicate/Duplicate MoS2 67.tif" #define filepath
refImage = cv2.imread(ref_filename) 

#Get contours and zernike moments of the reference flake
(refcnts, refZernikeMoment) = get_moments(ref_filename)

#Define empty lists to store results
image_name_list = []
zernike_distance_list = []
size_list = []
width_list = []
area_list = []
aspect_ratio_list = []
zernike_list = []

#Apply each function to each image in a defined folder of genuine nanotags
for filename in glob.glob('MoS2 Nanotags/*.tif'):
    
    #trim filename from filepath, append to list removing extension (.tif)
    head, tail = os.path.split(filename)
    image_name_list.append(tail[:-4])
    
    #get contours and Zernike moments from image
    (cnts, zernikeMoments) = get_moments(filename)
    zernike_list.append(zernikeMoments) #append moments to results list

    #compute zernike moment distances
    zmd = dist.cdist(refZernikeMoment, zernikeMoments) #calculate Euclidean distance between reference image moments and current image moments
    zernike_distance_list.append(zmd[0,0])
    
    #measure A, L, W, L/W from flake contour
    area, lateral_size, width, ratio = measure_size(cnts)

    #append all results to lists
    area_list.append(area)
    size_list.append(lateral_size)
    width_list.append(width)
    aspect_ratio_list.append(ratio)

#find index of lowest ZMD i.e. most similar genuine flake to the reference image   
j = np.argmin(zernike_distance_list)
print(f"The lowest zernike distance match to the scanned flake is: {image_name_list[j]}") #print most similar genuine imaage name to terminal
print(f"The zernike moment distance of this pair is: {zernike_distance_list[j]}") #print ZMD score to terminal

#make decision
if zernike_distance_list[j] < 0.0315: #if the zernike distance is below the threshold (see manuscript)          
    print("The scanned flake is genuine.") #print decision to terminal
else: print("Sorry, the scanned flake is not in our dataset.") #print decision to terminal

#Load the most similar genuine image to the reference image
match = cv2.imread(f"{head}/{image_name_list[j]}.tif")

#define figure with set rows and columns
fig = plt.figure(figsize=(6,3))
rows = 1
columns = 3

fig.add_subplot(rows, columns, 1) #to position 1 in the figure
plt.imshow(cv2.cvtColor(refImage, cv2.COLOR_BGR2RGB)) #load reference image, convert BGR to RGB to ensure correct colour in cv2.
plt.axis('off') #turn off axis
plt.title("Reference Image") #add title

fig.add_subplot(rows, columns, 2) #to position 2 in the figure
plt.imshow(cv2.cvtColor(match, cv2.COLOR_BGR2RGB)) #load most similar image, convert BGR to RGB to ensure correct colour in cv2.
plt.axis('off') #turn off axis
plt.title("Most similar Flake") #add title

fig.add_subplot(rows, columns, 3) #to position 3 in the figure

#if reference matches a genuine nanosheet
if zernike_distance_list[j] < 0.0315:  
    blank = np.ones(match.shape) #add blank white image with same dimensions as matched image
    plt.imshow(blank) #add image to position 3
    plt.axis('off') #turn off axis
    plt.title("Flake is legitimate.") #add heading to show flake is legitmate.

#if reference does not match genuine nanosheet
else:
    blank = np.ones(match.shape) #add blank white image with same dimension as matched image
    plt.imshow(blank) #add image to position 3
    plt.axis('off') #turn off axis
    plt.title("Flake is counterfeit.") #add heading to show no match was found

plt.tight_layout() #use tight layout
plt.show() #show the final figure

#----------COMPUTE ZMD SCORE OF ALL UNIQUE PAIRS--------------------#
#only enable this code if all results are needed for the dataset

#define empty list to store zernike moment distance for each flake pairing
zernike_difference_list = []

#nested for loop ensuring every contour is compared with every contour and itself, but not repeated. e.g. 00, 01, 02... 11, 12, 13... 22, 23, 24...
for i in range(len(zernike_list)):
    zernike_diff = dist.cdist(zernike_list[i], zernike_list[i]) #calculate euclidean distance between moments of all pairs of identical images (ZMD = 0) e.g FLake 1 Flake 1
    zernike_difference_list.append(zernike_diff[0,0]) #append result to list

    for j in range(i+1, len(zernike_list)):
        zernike_diff = dist.cdist(zernike_list[i], zernike_list[j]) #calculate Euclidean distance between moments of all unique image pairs e.g Flake 1 FLake 2
        zernike_difference_list.append(zernike_diff[0,0])

#define list to store compared name strings
name_compare_list = [] 

#nested for loop ensuring we create new string from both image names compared, for each unique image pairing
for i in range(len(image_name_list)):
    names_compared = str(f"{image_name_list[i]} {image_name_list[i]}") #between every pair of identical images e.g FLake 1 Flake 1
    name_compare_list.append(names_compared)
    for j in range(i+1, len(image_name_list)):
        names_compared = str(f"{image_name_list[i]} {image_name_list[j]}") #between every unique pair of images e.g FLake 1 Flake 2
        name_compare_list.append(names_compared)


#--------------EXPORT FLAKE PARAMETERS TO .CSV FILE, ENABLE IF NEEDED-------------------#
#Create tab delimited .csv file tabulating flake morphological parameter data
headers = ["Flake Number", "Lateral Size / nm", "Width / nm", "Area / um^2", "L/W Aspect Ratio"] #define collumn headers
with open('MoS2 Flake Parameters.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(image_name_list, size_list, width_list, area_list, aspect_ratio_list)) #zip results lists and write row by row

#create tab delimiter .csv file tabulating ZMD scores and names
headers = ["Flakes compared", "ZMD Score"] #define collumn headers
with open('MoS2 Nanotag ZMD scores.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(name_compare_list, zernike_difference_list)) #zip results lists and write row by row
