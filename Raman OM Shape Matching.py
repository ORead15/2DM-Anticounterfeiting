#Script is used to get the contours of nanosheets from Raman mapping images and corresponding optical images
#which requires the raman roi contours to be rescaled to the same aspect ratio as the optical image contours.
#a reference optical image can be defined, and the corresponding Raman map with the same shape from the folder of images is returned.

#import modules
import glob
import cv2
import os
import mahotas
from scipy.spatial import distance as dist
import numpy as np
import csv

#define function to get binary roi from a filepath to an image
def get_binary_roi(filename):
    om_img = cv2.imread(filename) #read image
    gray = cv2.cvtColor(om_img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.5) #blur
    
    #determine if ..._INV parameter is used or not. Threshold Raman images require cv2.THRESH_BINARY, Opical images require cv2.THRESH_BINARY_INV
    if raman == True:
        param = cv2.THRESH_BINARY
    else:
        param = cv2.THRESH_BINARY_INV

    thresh = cv2.threshold(blurred, 0, 255, param| cv2.THRESH_OTSU)[1] #convert to binary

    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]
    
    #create an empty mask for the contour and draw it
    mask = np.zeros(om_img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)

    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]

    return (roi, cnts)

#define function to compute Zernike moments from the contours of a binary image
def get_moments(roi, cnts):
    zernikeMoments = [] #initialise list to store zernike moments
    
    #compute Zernike Moments from roi, add to list.
    zMoments = mahotas.features.zernike_moments(roi, cv2.minEnclosingCircle(cnts)[1], degree=8)
    zernikeMoments.append(zMoments)

    return zernikeMoments

#define function to get the dimensions of an image
def get_OM_dimensions(image):
    roi_dimensions = image.shape #(height, width) = [x,y]
    roi_height = roi_dimensions[0]
    roi_width = roi_dimensions[1]
    return(roi_height, roi_width)

#define results lists
om_moments_list = []
om_imagename_list = []
om_filename_list = []
dimensions_list = []
raman_moments_list = []
raman_imagename_list = []
raman_filename_list = []

#global variable to switch between cv2.THRESH_BINARY (True) or cv2.THRESH_BINARY_INV (False)
raman = False  #Set global variable to switch to cv2.THRESH_BINARY_INV for optical image analysis

#loop through OM images extracting moments, names, dimensions
for filename in glob.glob("D:\Anticounterfeiting Paper\POC Study\Flake Library\Raman Shapes\OM resave/*.tif"):
    #trim image name from filepath
    head, tail = os.path.split(filename)
    image_name = tail[:-4]

    om_roi, om_cnts = get_binary_roi(filename) # get ROI and contours
    
    om_moments = get_moments(om_roi, om_cnts) #get moments

    om_moments_list.append(om_moments) # store moments
    
    om_imagename_list.append(f"{image_name} om") # store optical image name

    om_height, om_width = get_OM_dimensions(om_roi) #get dimensions
    dimensions = (om_width, om_height) #re-order for use with cv2.resize
    dimensions_list.append(dimensions) #store dimensions
    om_filename_list.append(filename) #store filename

raman = True #change global variable to switch to cv2.THRESH_BINARY for Raman map analysis

z = 0 #initialise count

for filename in glob.glob("D:\Anticounterfeiting Paper\POC Study\Flake Library\Raman Shapes\Rescaled Raman Pad/*.tif"):
    #trim image name from filepath
    head, tail = os.path.split(filename)
    image_name = tail[:-4]
    
    raman_roi, raman_cnts = get_binary_roi(filename) #get ROI and contours

    raman_roi_resize = cv2.resize(raman_roi, dimensions_list[z], interpolation= cv2.INTER_AREA) #resize raman roi to OM roi using OM image dimensions accessed from list.
    
    #find contours, sort by area, grab largest contour
    raman_cnts, hierarchy = cv2.findContours(raman_roi_resize.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(raman_cnts, key=cv2.contourArea, reverse = True)
    raman_cnts = sorted_cnts[0]

    raman_moments = get_moments(raman_roi_resize, raman_cnts) #get moments of resized raman
    
    raman_moments_list.append(raman_moments) #store moments
    raman_imagename_list.append(f"{image_name} raman") #store raman image name
    raman_filename_list.append(filename) #store filename

    z += 1 #increment count to get next dimension from list on next iteration
    
ZMD_list = [] #define ZMD list

#loop through every unique combination of moment pairs and calculate ZMD using Euclidean distance between two sets of moments
#i.e raman 1 om 1, raman 1 om 2, raman 2 om 1, raman 2 om 2 etc...
for i in range(len(raman_moments_list)):
    for j in range(0, len(raman_moments_list)):
        ZMD = dist.cdist(raman_moments_list[i], om_moments_list[j])   #calculate ZMD using Euclidean distance between two sets of moments
        ZMD_list.append(ZMD[0,0])

names_compared_list = [] #initialise lsit to store compared names

#loop through every unique combination of names and combine strings
for i in range(0, len(raman_imagename_list)):
    for j in range(0, len(raman_imagename_list)):
        names_compared = str(f"{raman_imagename_list[i]} {om_imagename_list[j]}")
        names_compared_list.append(names_compared)


#export names compared and corresponding ZMD score as tab delimited .csv file
headers = ["Flakes compared", "Zernike Moment Difference"] #define column headers
with open('OM Raman ZMD scores.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(names_compared_list, ZMD_list))

#-----------------DEFINE OPTICAL/RAMAN IMAGE TO FIND RAMAN/OPTICAL SHAPE MATCH-------------#

#Define file path to reference image to check for authenticity
filepath = "D:\Anticounterfeiting Paper\POC Study\Flake Library\Raman Shapes\OM resave/Flake 1.tif"

raman = False #change to True if reference is a Raman map, False if OM image

ref_roi, ref_cnts = get_binary_roi(filepath) #get reference roi and contours

ref_moments = get_moments(ref_roi, ref_cnts) #get reference moments

ref_raman_moments_list = [] #define moments list

#compute ZMD between reference and all raman images
for i in range(0, len(raman_moments_list)):
    ZMD = dist.cdist(ref_moments, raman_moments_list[i]) #calculate ZMD using Euclidean distance between two sets of moments
    ref_raman_moments_list.append(ZMD[0,0])

closest_index = np.argmin(ref_raman_moments_list) #return the index of lowest ZMD score (most similar shape match)
closest_raman_match = cv2.imread(raman_filename_list[closest_index]) #load image of closest match

ref_om_moments_list = [] #define moments list

#compute ZMD between reference and all optical images
for i in range(0, len(raman_moments_list)):
    ZMD = dist.cdist(ref_moments, om_moments_list[i]) #calculate ZMD using Euclidean distance between two sets of moments
    ref_om_moments_list.append(ZMD[0,0])

closest_index = np.argmin(ref_om_moments_list) #return the index of lowest ZMD score
closest_om_match = cv2.imread(om_filename_list[closest_index]) #load image of closest match


print(f"The best Raman match to the reference is: {raman_imagename_list[closest_index]}")
print(f"The ZMD value is: {ref_raman_moments_list[closest_index]}")
cv2.imshow("Closest Raman match", closest_raman_match) #show image
cv2.waitKey(0)

print(f"The best optical match to the reference is: {om_imagename_list[closest_index]}")
print(f"The ZMD value is: {ref_om_moments_list[closest_index]}")
cv2.imshow("Closest OM match", closest_om_match) #show image
cv2.waitKey(0)














# #this loop is needed to resize the raman roi to be the same as the om images so moments can be extracted and compared
# resized_img_list = []
# i = 0
# resized_raman_moments = []
# for om_thresh in om_image_list:
#     zern_moms = []
#     h,w = get_OM_dimensions(om_thresh)
#     dims = (w, h)
#     resized_img = cv2.resize(raman_image_list[i], dims, interpolation= cv2.INTER_AREA)
    

#     cnts, hierarchy = cv2.findContours(resized_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
#     cnts = sorted_cnts[0]

#     zMoments = mahotas.features.zernike_moments(resized_img, cv2.minEnclosingCircle(cnts)[1], degree=8)
#     zern_moms.append(zMoments)

#     resized_raman_moments.append(zern_moms)

#     resized_img_list.append(resized_img)

#     cv2.imwrite("D:\Anticounterfeiting Paper\POC Study\Flake Library\Raman Shapes\Raman Contours/" + om_samplename_list[i] + ".tif", resized_img)
#     #cv2.imshow("resize", resized_img)
#     #cv2.waitKey(0)
#     i += 1
    
# #define empty list to store zernike distance difference for each flake pairing
# zernike_distance_list = []

# for i in range(len(resized_raman_moments)):
#     #calculate zernike moment distance between OM and Raman images
#     zernike_diff = dist.cdist(resized_raman_moments[i], om_moments_list[i])   #00 11 22 33 44 ...
#     zernike_distance_list.append(zernike_diff[0,0])
    

#     for j in range(i+1, len(resized_raman_moments)):
#         zernike_diff = dist.cdist(resized_raman_moments[i], om_moments_list[j])   #01 02 03..., 12 13 14..., 23 24 25.., etc.
#         zernike_distance_list.append(zernike_diff[0,0])
        

# #Define file path to Raman map to check for authenticity
# filepath = "D:\Anticounterfeiting Paper\POC Study\Flake Library\Raman Shapes\Reference/Flake 12.png"

# #Get contours and zernike moments of the referene flake
# (refcnts, refZernikeMoments, roi) = get_moments_raman(filepath)

# #print(refZernikeMoments)
# #print(resized_raman_moments[0])

# reference_moments_list = []

# for i in range(len(om_moments_list)):
#     zern_diff = dist.cdist(refZernikeMoments, om_moments_list[i])
#     reference_moments_list.append(zern_diff[0,0])

# print(reference_moments_list)

# closest_index = np.argmin(reference_moments_list)
# closest_match = cv2.imread(om_filename_list[closest_index])

# print("best match to reference is:", om_samplename_list[closest_index])
# print("The ZMD value is:", reference_moments_list[closest_index])

# cv2.imshow("closest match", closest_match)
# cv2.waitKey(0)
