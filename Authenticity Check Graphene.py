#import modules
import cv2 #pip install opencv-python
import os
import mahotas #pip install mahotas
import numpy as np #pip install numpy
import glob
from scipy.spatial import distance as dist #pip install scipy
from matplotlib import pyplot as plt #pip install matplotlib
import csv

#define function to get zernike moments of a nanosheet contour from a cropped image of a nanosheet
def get_moments(image):
    zernikeMoments = [] #initialise list to store zernike moments
  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.5) #blur
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #convert to binary threshold image
    
    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]
    
    #create an empty mask for the contour and draw filled contour.
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)
  
    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]

    #compute Zernike Moments from roi, add to list.
    zMoments = mahotas.features.zernike_moments(mask, cv2.minEnclosingCircle(cnts)[1], degree=8) #use min enclosing circle of contour as radius, calculate moments up to order 8.
    zMoments = zMoments[2:] #exclude first 2 z moments for translation invariance
    zernikeMoments.append(zMoments)
       
    #return nanosheet contour and its Zernike moments
    return (cnts, zernikeMoments)

#define function to return L, W, A and L/W from a nanosheet contour
def measure_size(flake_contour):
    #calculate pixel size and area
    pixel_size = 1.0 / image_scale # in um
    pixel_area = pixel_size**2 # in um^2
    
    #calculate flake area in pixels and convert to um^2.
    flake_area = cv2.contourArea(flake_contour) #counts all pixels in contour
    flake_area = flake_area * pixel_area #convert units to um^2

    #Calculate flake L, W and L/W
    bounding_rect = cv2.minAreaRect(flake_contour) #create bounding rectangle with minimum area around contour
    (x,y), (w,h), angle = bounding_rect #x,y centre points of rectangle, w h, width and height of rectangle in pixels
    
    #take largest bounding dimension as L and the smaller as W
    if h > w:
        lateral_size = h
        width = w
    else: 
        lateral_size = w 
        width = h
    lateral_size = lateral_size * pixel_size #convert pixels to um
    width = width * pixel_size #converts pixels to um
    
    ratio = lateral_size / width #compute aspect ratio

    return flake_area, lateral_size, width, ratio


image_scale = 12.9 #set image scale pixels/um (measure from scale bar)

#Define Reference Image i.e. nanosheet to check for authenticity
ref_filename = "Graphene Nanotag Duplicates/Duplicate Graphene 2.tif" #define filepath to image e.g in duplicate nanotags folder
refImage = cv2.imread(ref_filename) #read image

(refcnts, refZernikeMoments) = get_moments(refImage) #Get contours and zernike moments from the reference nanosheet

#Define empty lists to store flake parameters
image_name_list = []
zernike_distance_list = []
size_list = []
width_list = []
area_list = []
aspect_ratio_list = []
zernike_list = []

#Apply each function to each image in a defined folder containing 'genuine' graphene nanotags
for filename in glob.glob('Graphene Nanotags/*.tif'):
    
    #trim image name from filepath, append to list removing extension (.tif)
    head, tail = os.path.split(filename)
    image_name_list.append(tail[:-4])
    
    #read image and pass it through get_moments function
    image = cv2.imread(filename)
    (cnts, zernikeMoments) = get_moments(image) #get contours and Zernike moments
    zernike_list.append(zernikeMoments) #append moments to list

    #compute Euclidean distance between reference and image moments
    zmd = dist.cdist(refZernikeMoments, zernikeMoments)
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
print(f"The lowest zernike distance match to the scanned flake is: {image_name_list[j]}") #print cloest match from genuine data set to terminal
print(f"The zernike moment distance of this pair is: {zernike_distance_list[j]}") #print its ZMD score

#print authenticity decision to terminal
if zernike_distance_list[j] < 0.0315:  #if the zernike distance is below the ZMD threshold (see manuscript)      
    print("The scanned flake is genuine.") 
else: print("Sorry, the scanned flake is not in our dataset.") #if not

#Load the flake image from genuine data set that is the closest match to the reference image.
match = cv2.imread(f"{head}/{image_name_list[j]}.tif")

#define figure with defined rows and columns
fig = plt.figure(figsize=(6,3))
rows = 1
columns = 3

#reference
fig.add_subplot(rows, columns, 1) #to position 1 in the figure
plt.imshow(cv2.cvtColor(refImage, cv2.COLOR_BGR2RGB)) #load reference image, convert BGR to RGB to ensure correct colour in cv2.
plt.axis('off') #turn off axis
plt.title("Reference Image") #add title

#match
fig.add_subplot(rows, columns, 2) #to position 2 in the figure
plt.imshow(cv2.cvtColor(match, cv2.COLOR_BGR2RGB)) #load closest match image, convert BGR to RGB to ensure correct colour in cv2.
plt.axis('off') #turn off axis
plt.title("Most similar Flake") #add title

#decision
fig.add_subplot(rows, columns, 3) #to position 3 in the figure

#if reference matches a genuine nanosheet
if zernike_distance_list[j] < 0.0315:  
    blank = np.ones(match.shape) #add blank white image with same dimensions as matched image
    plt.imshow(blank) #add image to position 3
    plt.axis('off') #turn off axis
    plt.title("Flake is legitimate.") #add heading to show flake is legitmate.

#if reference does not match a genuine nanosheet
else:
    blank = np.ones(match.shape) #add blank white image with same dimension as matched image
    plt.imshow(blank) #add image to position 3
    plt.axis('off') #turn off axis
    plt.title("Flake is counterfeit.") #add heading to show no match was found

plt.tight_layout() #use tight layout
plt.show() #show the final figure


#---------------COMPUTE ZMD SCORE OF ALL UNIQUE IMAGE PAIRS----------------#
#enable if you wish to check all ZMD scores to check for false positive/negative matching.

#define empty list to store zernike distance difference for each flake pairing
zernike_difference_list = []

#nested for loop ensuring every contour is compared with every contour and itself, but not repeated. e.g. 00, 01, 02... 11, 12, 13... 22, 23, 24...
for i in range(len(zernike_list)):
    zernike_diff = dist.cdist(zernike_list[i], zernike_list[i]) #compute Euclidean diference between moments of identical images (ZMD = 0)
    zernike_difference_list.append(zernike_diff[0,0]) #append to list

    for j in range(i+1, len(zernike_list)): 
        zernike_diff = dist.cdist(zernike_list[i], zernike_list[j]) #compute Euclidean distance between moments of every unique image pairing with no duplicates.
        zernike_difference_list.append(zernike_diff[0,0]) #append to results list

#define list to store compared names
name_compare_list = [] 

#nested for loop to concatenate a string from both image names compared for each unique image pairing
for i in range(len(image_name_list)):
    names_compared = str(f"{image_name_list[i]} {image_name_list[i]}") #concatenate string of identical images (e.g flake 1 flake 1)
    name_compare_list.append(names_compared) #append to results list
    for j in range(i+1, len(image_name_list)):
        names_compared = str(f"{image_name_list[i]} {image_name_list[j]}") #concatenate string of unique image pairs (e.g flake 1 flake 2)
        name_compare_list.append(names_compared)


#--------------EXPORT NAMES AND ZMD SCORES TO .CSV FILE-------------------#
#create tab delimited .csv file tabulating ZMD scores and names.
headers = ["Flakes compared", "Zernike Moment Distance"] #define collumn headers
with open('Graphene nanotag ZMD Scores.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(name_compare_list, zernike_difference_list)) #zip results lists and write row by row.

#--------------EXPORT FLAKE PARAMETERS TO .CSV FILE-------------------#
#Create .csv file tabulating flake morphological parameter data
headers = ["Image Name", "Lateral Size / um", "Width / um", "Area / um^2", "L/W Aspect Ratio"] #define collumn headers
with open('Graphene nanotag morphology.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(image_name_list, size_list, width_list, area_list, aspect_ratio_list)) #zip results lists and write row by row.
