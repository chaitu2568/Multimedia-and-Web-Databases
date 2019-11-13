import cv2
import numpy as np
import pandas as pd
import glob
import scipy
import matplotlib.pyplot as plt
from scipy.stats import skew
import skimage
from skimage import feature
import pickle
import operator
import math

# Function Splits the given image into 100*100 windows and returns the final vector and corresponding
# heights and widhts of final image vector
def img_to_grids(image):
    height=image.shape[0]
    h=height//100
    width=image.shape[1]
    w=width//100
    final=np.zeros(shape=(h,w),dtype=object)
    start_row=0
    end_row=100
    for i in range(0,h):
        start_col=0
        end_col=100
        for j in range(0,w):
            final[i][j]=image[start_row:end_row,start_col:end_col]
            start_col+=100
            end_col+=100
        start_row+=100
        end_row+=100
    return final,h,w

# Function for calculating the color Moments
def COLOR_MOMENTS(image):
#     Function for converting image to YUV color Model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    window_img,h,w=img_to_grids(image)
    vector=[]
    for col in range(window_img.shape[1]):
        for row in range(window_img.shape[0]):
            window_moments=[]
#         Calculating First Momemt
            means=np.mean(window_img[row][col],axis=0)
            moment1=np.mean(means,axis=0)
            window_moments.extend(moment1)
#         Calculating Second Moment
            st_dev=np.std(window_img[row][col],axis=0)
            moment2=np.std(st_dev,axis=0)
            window_moments.extend(moment2)
#         Calculating Third Moment
            skew_ness=skew(window_img[row][col],axis=0)
            moment3=skew(skew_ness,axis=0)
            window_moments.extend(moment3)
#             Concatinating the list of moments values for each window to final vector
            vector.extend(window_moments)
    vector=np.array(vector)
#     returns the feature descriptor for COLOR MOMENTS in the form of 1D-np Array
    return vector

# Function for calculating Local Binary Patterns
def LBP(image):
#     Function for converting the image to GRAY_Scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     Calling Function 'img_to_grids' to split the image to 100*100 windows
    window_img,h,w=img_to_grids(image)
    final=[]
    for col in range(window_img.shape[1]):
        for row in range(window_img.shape[0]):
            near_points=8
            radius=2
#             Calculating LBP which gives the matrix of binary numbers for each window in similar dimensions
            local_B_pat = feature.local_binary_pattern(window_img[row][col], near_points,radius, method="uniform")
#     Calculating the histogram for each window
            (hist, _) = np.histogram(local_B_pat.ravel(),bins=np.arange(0,near_points + 3),range=(0, near_points + 2))
            final.extend(hist)
    final=np.array(final)
#     returns the feature descriptor for LOCAL BINARY PATTERNS in the form of 1D-np Array
    return final

# Main Function which returns the Feature descriptor, given path to image(IMAGE-ID) and model
def main_fun(image_id,model):
    image_id=cv2.imread(image_id)
    if model==1:
        vector=COLOR_MOMENTS(image_id)
    else:
        vector=LBP(image_id)
    return vector

# Function to store feature descriptors and corresponding Image_ID's for a given model to CSV file
def storing_features(folder,model):
    imageID=[]
    for path in folder:
        imageID.append(path[path.rfind("Hand"):-4])
    dics={}
    for i in range(len(imageID)):
        # APPENDING ALL THE FEATURES TO DATA FRAME
        dics[imageID[i]] = main_fun(folder[i],model)
#         df=df.append({'IMAGE_ID':folder[i],'FEATURE_DESCRIPTOR':main_fun(folder[i],model)},ignore_index=True)

    if model==1:
        file_name = "CM_Features.pickle"
    else:
        file_name = "LBP_Features.pickle"
    with open(file_name, 'wb') as handle:
        pickle.dump(dics, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Extracting all the image paths in a given folder to list called 'Filenames'
image_path=input('Please give the path containing folder of images:')
filenames = glob.glob(str(image_path)+"/*.jpg")
storing_features(filenames,1)
storing_features(filenames,2)


