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
#   Concatinating the list of moments values for each window to final vector
            vector.extend(window_moments)
    vector=np.array(vector)
#     returns the feature descriptor for COLOR MOMENTS in the form of 1D-np Array
    return vector

# Function for calculating Local Binary Patterns
def LBP(image):
#   Function for converting the image to GRAY_Scale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   Calling Function 'img_to_grids' to split the image to 100*100 windows
    window_img,h,w=img_to_grids(image)
    final=[]
    for col in range(window_img.shape[1]):
        for row in range(window_img.shape[0]):
            near_points=8
            radius=2
#   Calculating LBP which gives the matrix of binary numbers for each window in similar dimensions
            local_B_pat = feature.local_binary_pattern(window_img[row][col], near_points,radius, method="uniform")
#   Calculating the histogram for each window
            (hist, _) = np.histogram(local_B_pat.ravel(),bins=np.arange(0,near_points + 3),range=(0, near_points + 2))
            final.extend(hist)
    final=np.array(final)
#   returns the feature descriptor for LOCAL BINARY PATTERNS in the form of 1D-np Array
    return final

# Main Function which returns the Feature descriptor, given path to image(IMAGE-ID) and model
def main_fun(image_id,model):
    image_id=cv2.imread(image_id)
    if model==1:
        vector=COLOR_MOMENTS(image_id)
    else:
        vector=LBP(image_id)
    return vector

def cosine_similarity(a, b):
    return sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))

def eucledian(a, b):
    sum = 0
    for i in range(len(a)):
        sum += math.pow(a[i] - b[i], 2)
    return math.sqrt(sum)

def similarity_measure(image_path,model,k):
    # Please give the path to test-image folder
    test_images_folderpath=input('Please give the path to test-image folder')
# READING THE GIVEN IMAGEID PATH AND CALCULATING CORRESPONDING FEATURE DESCRIPTOR
    if model==1:
        test_vector=main_fun(image_path,1)
        file_name = "CM_Features.pickle"
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)
    else:
        file_name = "LBP_Features.pickle"
        test_vector=main_fun(image_path,2)
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)

    distances={}
    i=1
    print(test_vector.shape)
    for imageid,feature in b.items():
        print(feature.shape)
# CALCULATING COSINE SIMILARITY B/W GIVEN IMAGE AND ALL OTHER IMAGES IN GIVEN FOLDER
        distances[imageid]=cosine_similarity(test_vector,feature)

# HASH TABLE CONTAINING IMAGES SORTED BY SIMILARITY IN DECREASING ORDER
    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1),reverse=True)
    similarity_ranking={}
# CREATING HASH-TABLE FOR RETURNING TOP 'K' IMAGES AND THEIR MATCHING/SIMILARITY SCORES
    for tup in sorted_distances:
        image=cv2.imread(str(test_images_folderpath)+"/" + tup[0] + ".jpg")
        image = cv2.resize(image,(240,240))
        cv2.imshow('Similar_image',image)
        cv2.waitKey()
        similarity_ranking[tup[0]]=tup[1]
        i+=1
        if i==k+1:
            break
    cv2.destroyAllWindows()
    return similarity_ranking

# Please Enter query image Path
query_image_path=input('Please give the path to query image path')

# # TAKES AN INPUT IMAGEID FROM THE USER
# image_id=input("Enter the Image ID:")

# Takes the model to which similarity measure is calculated
model_num=input("Enter the model number 1 for Color Moments and any number for Local Binary patterns:")

# Enter the K_value
k=input("Please give k value:")


# SETTING THE IMAGE PATH for query image
image_path = query_image_path
if int(model_num)==1:
    output=similarity_measure(image_path,1,int(k))
else:
    output=similarity_measure(image_path,2,int(k))
cv2.destroyAllWindows()
print('Similarity Rankings for k images')
print(output)

