#import modules
import numpy as np #pip install numpy
import mahotas #pip install mahotas
import cv2 #pip install opencv-python
from scipy.spatial import distance as dist #pip install scipy
import glob
import os
import csv

#function to return the thresholded image from a filepath to an optical image
def get_threshold(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.5) #blur
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #convert to binary    
    return thresh

#function to return the contours from a threshold image
def get_contours(threshold_image):
    cnts, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]
    return cnts

#function to resize an image using scaling factor
def resize_image(image,alpha):
    thresh_resize = cv2.resize(image, None, fx = alpha, fy = alpha, interpolation = cv2.INTER_LINEAR) #nearest interpolation works best
    return thresh_resize

#function to return zernike moments from a threshold image and the shape contours.
def get_moments(thresh_resize, cnts_resize):
    zernike_moments = []
    z_Moments = mahotas.features.zernike_moments(thresh_resize, cv2.minEnclosingCircle(cnts_resize)[1], degree=8)
    zernike_moments.append(z_Moments)
    return zernike_moments

#function to return the roi of a threshold by cropping bounding box of shape contours
def get_roi(thresh_image, cnts):
    #create an empty mask for the contour and draw filled contour.
    mask = np.zeros(thresh_image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)
  
    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]
    return roi


#scale invariance: 
#(1): m'00 = beta,  (2): let a = (Beta/m00)**0.5
#substitute (2) into (1): to get (3): m'00 = a**2m00 = Beta 
#Transform image function f(x,y) -> f(x/a, y/a) where a = (Beta/m00)**0.5

beta = 12000 # defined Beta value, desired value of m'00
zernike_moment_list = [] #define list to store moments
image_name_list = [] #define list to store image names

for filename in glob.glob("path_to_images/*.tif"):
    #trim filename from filepath, append to list removing extension (.tif)
    head, tail = os.path.split(filename)
    image_name = tail[:-4]
    image_name_list.append(image_name)

    #To acheive scale invariance
    thresh = get_threshold(filename) #get threshold image
    cv2.imwrite(f"{image_name}.tif", thresh) #save threshold image image
    cnts = get_contours(thresh) #get contours of threshold image
    M_1 = cv2.moments(cnts) #get image moments from contours
    m00 = int(M_1['m00']) #get zeroth order image moment (or use m00 = cv2.contourArea(cnts))
    alpha = (beta / m00)**0.5 #calculate scaling factor alpha
    thresh_resize = resize_image(thresh, alpha) #resize image using alpha
    cv2.imwrite(f"{image_name} resize.tif", thresh_resize) #save rescale image
    cnts_resize = get_contours(thresh_resize) #get contours from resized image

    #crop roi to get translation invariance                                                                                                         
    thresh_resized_roi = get_roi(thresh_resize, cnts_resize) #crop roi from normalised image
    cv2.imwrite(f"{image_name} scale normalized roi.tif", thresh_resized_roi) #save rescaled and centered image

    #calculate zernike moments from normalised roi
    zernike_moments = get_moments(thresh_resized_roi, cnts_resize) #calculate zernike moments from new image
    zernike_moment_list.append(zernike_moments) #store moments in list

#define empty list to store zernike moment distance difference for each flake pairing
zernike_difference_list = []

#nested for loop ensuring every contour is compared with every contour and itself, but not repeated. e.g. 00, 01, 02... 11, 12, 13... 22, 23, 24...
for i in range(len(zernike_moment_list)):
    zernike_diff = dist.cdist(zernike_moment_list[i], zernike_moment_list[i])
    zernike_difference_list.append(zernike_diff[0,0])

    for j in range(i+1, len(zernike_moment_list)):
        zernike_diff = dist.cdist(zernike_moment_list[i], zernike_moment_list[j])
        zernike_difference_list.append(zernike_diff[0,0])

#define list to store compared names
name_compare_list = [] 

#nested for loop ensuring we create new string from both image names compared, for each image pairing
for i in range(len(image_name_list)):
    names_compared = str(f"{image_name_list[i]} {image_name_list[i]}")
    name_compare_list.append(names_compared)
    for j in range(i+1, len(image_name_list)):
        names_compared = str(f"{image_name_list[i]} {image_name_list[j]}")
        name_compare_list.append(names_compared)


#--------------EXPORT NAMES AND ZMD SCORES TO .CSV FILE-------------------#
#create tab delimited .csv file tabulating ZMD scores and names.
headers = ["Flakes compared", "Zernike Distance Difference"] #define collumn headers
with open('Scale Normalised all data and dupes for SI.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(name_compare_list, zernike_difference_list))
