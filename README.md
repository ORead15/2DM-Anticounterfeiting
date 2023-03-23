# Anticounterfeiting Solution Based on the shape of 2D Materials produced by ECE.
*Oliver Read, Matthew Boyes, Xiuju Song, Jingjing Wang, Gianluca Fiora, Cinzia Casiraghi.

We propose individual 2D material nanosheets can act as identification tags by exploiting their unique shape using image processing and computer vision techniques. We provide here a series of scripts that can be used to replicate the results shown in the main manuscript and supporting information, or to be adapted and used with your own data sets. We have also provied the image datasets and raw spectroscopic data used in the manuscript. We recommend downloading the repository as a .zip file for ease of use.

Required Python libraries:
* NumPy
* OpenCv
* Mahotas
* SciPy
* MatPlotLib
* RamPy

Authenticity Check Steps:
1. Open 'Authenticity Check Graphene.py'
2. Define ref_filename (line 71): filepath to a reference image of a nanosheet you wish to check for authenticity. We recommend using any of the images in 'Graphene Nanotag Duplicates' or 'Counterfeit Graphene'.
3. Define the image_scale (line 68). For our data set use 12.9 px/um.
4. Ensure the folder containing 'genuine' nanosheet images is defined (line 86). We recommend using the folder 'Graphene Nanotags' which is supplied.
5. Run the script.

In summary:
* Each image in the folder (and the ref_image) is processed to obtain the thresholded binary image.
* The contour of the nanosheet is calculated.
* The region of interest (roi) is cropped using a bounding box of the contour. (achieving translation invariance)
* The Zernike moments are calculated from the binary roi up to order 8, using the radius as the minEnclosingCircle of the contour.
* The Euclidean distance is calculated between the Zernike Moments of the ref_image and every image in the folder. We refer to this as the Zernike Moment Distance (ZMD).
* The image pairing with the lowest ZMD is returned. If the ZMD is below the thresold of 0.0315 the two images are considered identical. If the ZMD is above the threshold the two images are considered unique.
* The script allows for rotational and translational invariant shape matching between images. It is also demonstrated to partially work with some scale difference between images. An alternative script 'Scale invariant Zernike Moment modification.py' is supplied which achieves translational, rotation and scale invariant shape matching.
* The script will output a crude figure in matplotlib showing the reference image, the most similar image from the genuine dataset and a decision on reference nanotag authenticity. 
* If lines 161-204 are enabled two .csv files will also be output. One containing the morphological parameters L, W, A and L/W for the nanosheets in the folder. The second containing the ZMD scores of every unique shape pairing in the folder of images. By opening the second you can sort the ZMD scores from low to high to check for false positive or false negative matching. The values from these two .csv files are used to produce the histograms shown in the manuscript.
* Note that an additional script 'Authenticity Check MoS2.py' is supplied for use with the corresponding MoS2 images and is ran following the same steps. This script features a slight variation as erosion and dilation of the threshold images is required to isolate the nanosheet contours for our data set.

Generating Streamline Raman Map steps:
1. Open 'Graphene Streamline Raman map area.py'.
2. Define filename (line 35) as filepath to a mapping data .txt file supplied in 'Streamline Raman data'.
3. Ensure output_folder_map and output_folder_thresh are defined as the filepaths to corresponding  folders in 'Raman Maps and OM images'.
4. Run script.
5. Repeat for all raw mapping data .txt files.

In summary:
* Script will produce mapping images based on the area under the spectra of the D, G and D' peaks of graphene.
* The raman mapping image will be saved to 'Raman Maps and OM Images/Raman area maps' folder.
* The thresholded raman mapping image will be saved to 'Raman Maps and OM Images/Raman threshold' folder.


Raman-Optical Shape Matching Steps:
First we need to resize the Raman maps so the nanosheet has the same aspect ratio as the corresponding optical imaage.
1. Open 'Resize Raman contours to OM contours.py'.
2. Ensure your filepath to 'Raman Maps and OM Images\Raman Threshold/*.tif' is specified in the for loop (line 64)
3. Ensure your filepath to 'Raman Maps and OM Images\OM Raman/*.tif' is specified in the for loop (line 79)
4. Ensure your filepath to the corresponding folders on Lines 71, 75, 82, 91, 95 are specified.
5. Run script.

We can then run the next script.
1. Open 'Raman OM Shape Matching.py'
2. Ensure your filepath to 'Raman Maps and OM Images\OM resave/*.tif' is specified in the for loop (line 73)
3. Ensure your filepath to 'Raman Maps and OM Images\Rescaled Raman Pad/*.tif' is specified in the for loop (line 95)
4. Define a filepath to a reference optical image from the OM Resave folder. (line 146)
5. Run script.

In summary:
* bla bla bla
* bla bla bla
* bla bla bla

PL and Raman fitting scripts:
1. Open 'MoS2 Bulk PL Spectra fitting.py.
2. Set file path to 'PL MoS2 Data' folder in for loop (line 25).
3. Run Script.
4. Open 'MoS2 Bulk Raman Spectra fitting.py.
5. Set file path to 'Raman MoS2 Data' folder in for loop (line 25).
6. Run script.




