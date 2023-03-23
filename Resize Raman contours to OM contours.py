#Script is used to scale the nanosheet shape from produced raman maps to the same aspect ratio as the corresponding optical image so
#that the edges of the nanosheet can be extracted from the raman map, and compared to the edges from the optical image using the next script.

#import modules
import cv2
import glob
import os
import numpy as np

#define function to return the bounding box coordinates of a nanosheet from a raman map image
def get_raman_roi(filename):
    img = cv2.imread(filename) #read image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 2.5) #blur
    img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #convert to binary

    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]

    #create an empty mask for the contour and draw filled contour
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)

    (x, y, w, h) = cv2.boundingRect(cnts) #get bounding box coords
    
    return(x, y, w, h)

#define function to get the nanosheet roi from an optical image of a nanosheet
def get_OM_shape(filename):
    #trim sample/file name from filepath and remove .txt extension
    head, tail = os.path.split(filename)
    image_name = tail[:-4]
    
    img = cv2.imread(filename) #read image
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    img_blur = cv2.GaussianBlur(img_gray,(7, 7), 2.5) #blur
    img_binary = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] #convert to binary
    
    #find contours, sort by area, grab largest contour
    cnts, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
    cnts = sorted_cnts[0]

    #create an empty mask for the contour and draw it
    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [cnts], -1, 255, -1)
    
    (x, y, w, h) = cv2.boundingRect(cnts) #get bounding box coords

    roi = mask[y:y + h, x:x + w] #crop roi from mask
   
    return(roi, image_name)

#define function to get dimensions of an image
def get_OM_dimensions(image):
    roi_dimensions = image.shape #(height, width) = [x,y]
    roi_height = roi_dimensions[0]
    roi_width = roi_dimensions[1]
    return(roi_height, roi_width)

#for each raman threshold map image in defined folder, get roi, use it to crop raman map and save it.
for filename in glob.glob('Raman Shapes\Raman Threshold/*.png'):
    #trim sample/file name from filepath and remove .txt extension
    head, tail = os.path.split(filename)
    image_name = tail[:-4]
    
    (x, y, w, h) = get_raman_roi(filename) #get bounding box coords of raman map threshold image
    
    raman_map = cv2.imread(f"Raman Shapes\Raman area maps/{image_name}.png") #load raman area map
   
    raman_roi = raman_map[y:y + h, x:x + w]  #crop roi from map
    
    cv2.imwrite(f"Raman Shapes\Raman ROI/{image_name}.tif", raman_roi) #save raman roi 

    
#loop through all OM images in defined folder, crop roi, find dimensions of roi, load raman map and scale it to OM size, pad and save.
for filename in glob.glob('Raman Shapes\OM Raman/*.tif'):
    om_img = cv2.imread(filename) #read image
    roi, image_name = get_OM_shape(filename) #get roi and image name
    cv2.imwrite(f"Raman Shapes\OM resave/{image_name}.tif", om_img) #save the om_image to prevent resizing errors when loading/reloading images with cv2, cause unknown to me but it appears the image dpi changes but size etc stays the same.

    height, width = get_OM_dimensions(roi) #get height and width of roi
    dimensions = (width, height) #reorder to use in cv2.resize
    
    raman_roi = cv2.imread(f"Raman Shapes\Raman ROI/{image_name}.tif") #load raman roi
     
    raman_rescale = cv2.resize(raman_roi, dimensions, interpolation= cv2.INTER_AREA) #rescale to aspect ratio of OM image
    
    cv2.imwrite(f"Raman Shapes\Rescaled raman/{image_name}.tif", raman_rescale) 

    raman_pad = cv2.copyMakeBorder(raman_rescale, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value = 0) #pad raman roi with 10 pixels so edges can be extracted

    cv2.imwrite(f"Raman Shapes\Rescaled Raman Pad/{image_name}.tif", raman_pad) #save rescale raman image
