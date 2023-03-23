import numpy as np
import mahotas
import cv2
from scipy.spatial import distance as dist
import scipy
import glob
import os
import csv

def get_threshold(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 2.5) #blur
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #convert to binary    

    return thresh

def get_contours(threshold_image):
    cnts, hierarchy = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]
    return cnts

def resize_image(image,scale):
    thresh_resize = cv2.resize(image, None, fx = scale, fy = scale, interpolation = cv2.INTER_LINEAR) #nearest interpolation works best
    return thresh_resize

def get_moments(thresh_resize, cnts_resize):
    zernike_moments = []
    z_Moments = mahotas.features.zernike_moments(thresh_resize, cv2.minEnclosingCircle(cnts_resize)[1], degree=8)
    zernike_moments.append(z_Moments)
    return zernike_moments

def get_roi(thresh_image, cnts):
    
    #create an empty mask for the contour and draw filled contour.
    mask = np.zeros(thresh_image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)
  
    #extract bounding rectangle of contour, use it to crop roi from mask.
    (x, y, w, h) = cv2.boundingRect(cnts)
    roi = mask[y:y + h, x:x + w]
    
    return roi

def get_centroid(cnts):
    
    moments = cv2.moments(cnts)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    
    return cX, cY

def center_image(image, cX, cY):
    rows, cols = image.shape

    M = np.float32([[1, 0, cols/2-cX], [0, 1, rows/2-cY]])
    centered_image = cv2.warpAffine(image, M, (cols, rows))

    return centered_image

#scale invariance: 
#(1): m'00 = beta,  (2): let a = (Beta/m00)**0.5
#substitute (2) into (1): to get (3): m'00 = a**2m00 = Beta 
#Transform image function f(x,y) -> f(x/a, y/a) where a = (Beta/m00)**0.5

beta = 12000 # defined Beta value

zernike_moment_list = [] #store moments in list
image_name_list = [] #store image names in list

for filename in glob.glob("E:\Anticounterfeiting Paper\Image Datasets\Graphene nanotags and duplicates/*.tif"):
    #trim filename from filepath, append to list removing extension (.tif)
    head, tail = os.path.split(filename)
    image_name = tail[:-4]
    image_name_list.append(image_name)

    #To acheive scale invariance
    thresh = get_threshold(filename) #get threshold image
    cv2.imwrite(f"{image_name}.tif", thresh) #save threshold image image
    cnts = get_contours(thresh) #get contours of threshold image
    area = cv2.contourArea(cnts) #get area of contours in pixels
    
    M_1 = cv2.moments(cnts) #get image moments from contours
    m00 = int(M_1['m00']) #get zeroth order image moment (or use cv2.contourArea(cnts))
    print(f"m00 is {m00}")

    alpha = (beta / m00)**0.5 #calculate scaling factor alpha
    print(f"alpha is {alpha}")

    thresh_resize = resize_image(thresh, alpha) #resize image using alpha
    cv2.imwrite(f"{image_name} resize.tif", thresh_resize) #save rescale image
    cnts_resize = get_contours(thresh_resize) #get contours from resized image

    #To acheive translation invariance
                                                                         
                                                                       #cX, cY = get_centroid(cnts_resize) #get centroid of contour
                                                                   #thresh_norm = center_image(thresh_resize, cX, cY) #centre contour in image
                                                                    #cv2.imwrite(f"{image_name} normalized.tif", thresh_norm) #save rescaled and centered image

    #compute checks
    M_2 = cv2.moments(cnts_resize)
    m00_2 = int(M_2['m00'])
    print(f"m00' is {m00_2}, and beta is {beta}") #should be the same or close

    #crop roi to get translation invariance
                                                                                                                    #cnts_norm = get_contours(thresh_norm) #get contours from normalised image
    thresh_resized_roi = get_roi(thresh_resize, cnts_resize) #crop roi from normalised image
    cv2.imwrite(f"{image_name} scale normalized roi.tif", thresh_resized_roi) #save rescaled and centered image

    #calculate zernike moments from normalised roi
    zernike_moments = get_moments(thresh_resized_roi, cnts_resize) #calculate zernike moments from new image
    zernike_moment_list.append(zernike_moments) #store moments in list


#---------------COMPARE ALL UNIQUE FLAKE PAIRINGS ENABLE IF NEEDED----------------#

#define empty list to store zernike distance difference for each flake pairing
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
#create tab delimiter .csv file tabulating ZMD scores and names.
headers = ["Flakes compared", "Zernike Distance Difference"] #define collumn headers
with open('Scale Normalised all data and dupes for SI.csv', 'w', newline = '') as file: #create new .csv tab delimited document with specified file name ensuring no gaps between rows
    writer = csv.writer(file, delimiter='\t')
    dw = csv.DictWriter(file, delimiter = '\t', fieldnames=headers)
    dw.writeheader()
    writer.writerows(zip(name_compare_list, zernike_difference_list))
