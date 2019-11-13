import os
import pandas as pd
import numpy as np
from T1 import pca, LDA, SVD, nmf
from sklearn.cluster import KMeans
from phase1combined import storing_features


''' Method which gives the k-latent semantics for a given meta-data information'''

def metadata_latent_semantics(folder_path, meta_data, drt, k, color_model):

    ''' meta-data info of the hands is read from 'HandInfo.csv' file which is present in current working directory '''

    df = pd.read_csv('HandInfo.csv')
    if meta_data == 'dorsal':
        df = df[df['aspectOfHand'].str.contains('dorsal')]
        images = df['imageName'].values

    elif meta_data == 'palmar':
        df = df[df['aspectOfHand'].str.contains('palmar')]
        images = df['imageName'].values

    elif meta_data == 'right':
        df = df[df['aspectOfHand'].str.contains('right')]
        images = df['imageName'].values

    elif meta_data == 'left':
        df = df[df['aspectOfHand'].str.contains('left')]
        images = df['imageName'].values

    elif meta_data == 'male':
        df = df[df['gender']=='male']
        images = df['imageName'].values

    elif meta_data == 'female':
        df = df[df['gender']=='female']
        images = df['imageName'].values

    elif meta_data == 'with_accessories':
        df = df[df['accessories']== 1]
        images = df['imageName'].values

    elif meta_data == 'without_accessories':
        df = df[df['accessories']== 0]
        images = df['imageName'].values

    images =images.tolist()
    fin_img = []
    for img in images:
        fin_img.append(img.replace('.jpg',''))
    file_name = folder_path + "_" + color_model + "_Features.csv"

    ''' Extracting the Features of given meta-data labels from a dataset containing features
        of all images'''
    if os.path.exists(file_name) == False:
        storing_features(test_folderpath=folder_path, model=color_model)

    df_final = pd.read_csv(file_name)
    df_final.rename(columns = {'Unnamed: 0':'imageid'}, inplace=True)
    df_final = df_final.loc[df_final['imageid'].isin(fin_img)]

    data = df_final.values
    data_matrix = data[:,1:]
    image_ids = data[:,0]

    sub_filename = folder_path + "_" + meta_data + "_" + color_model + "_" + drt + "_" + str(k) + ".csv"

    ''' Calculating the k-latenet semantics for the features of extracted labels'''
    if drt == 'PCA':
        comp, proj_mat,_ = pca(data_matrix, k)

    elif drt == 'SVD':
        comp, proj_mat,_ = SVD(data_matrix, k)

    elif drt == 'LDA':
        if color_model == 'CM':
            final_matrix = []
            for i in range(data_matrix.shape[0]):
                for j in range(0, data_matrix.shape[1], 9):
                    final_matrix.append(list(data_matrix[i, j:j+9]))
            final_matrix = np.array(final_matrix)
            kmeans = KMeans(n_clusters=400, random_state=0).fit(final_matrix)
            final_store_matrix = np.zeros((data_matrix.shape[0], 400))
            for i in range(data_matrix.shape[0]):
                start = i * data_matrix.shape[1]//9
                for j in range(data_matrix.shape[1]//9):
                    final_store_matrix[i, kmeans.labels_[start+j]] += 1
            data_matrix = final_store_matrix

        comp, proj_mat,_ = LDA(data_matrix, k)

    elif drt == 'NMF':

        if color_model == 'CM':
            final_matrix = []
            for i in range(data_matrix.shape[0]):
                for j in range(0, data_matrix.shape[1], 9):
                    final_matrix.append(list(data_matrix[i, j:j+9]))
            final_matrix = np.array(final_matrix)
            kmeans = KMeans(n_clusters=400, random_state=0).fit(final_matrix)
            final_store_matrix = np.zeros((data_matrix.shape[0], 400))
            for i in range(data_matrix.shape[0]):
                start = i * data_matrix.shape[1]//9
                for j in range(data_matrix.shape[1]//9):
                    final_store_matrix[i, kmeans.labels_[start+j]] += 1
            data_matrix = final_store_matrix
        comp, proj_mat,_ = nmf(data_matrix, k)

    proj_mat = proj_mat.tolist()
    image_ids = image_ids.tolist()
    cols = [color_model+"_"+drt+"_"+str(i) for i in range(1,k+1)]
    data_frame = pd.DataFrame(data=proj_mat,columns=cols,index=image_ids)

    if os.path.exists(sub_filename) == True:
        os.remove(sub_filename)
    data_frame.to_csv(sub_filename)

    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM'):
        return kmeans, comp, sub_filename

    return comp, sub_filename


if __name__ == '__main__':

    folder_path = input("enter the folder path for images")

    color_model = input("Enter the model'CM' for Color Moments\n"
                        "'LBP' for Local Binary patterns:\n"
                        " 'HOG' for Histogram of Gradients:\n"
                        " 'SIFT' for Scale In-variant feature Transform ")

    k = int(input("Please give k value to give latent semantics: "))

    label = input("Please give the label: ")

    dr_technique = input("Enter the dimensionality reduction technqiue:\n"
                        "give PCA: Principle Component Analysis\n "
                        "SVD: Singular Value Decompostion\n "
                        "LDA: Latent Dirichlet Allocation\n"
                        "NMF: Non-Negative Matrix Factorization ")

    if (dr_technique == 'LDA' and color_model=='CM') or (dr_technique == 'NMF' and color_model=='CM'):
        _, _, term_weight_name = metadata_latent_semantics(folder_path=folder_path, meta_data=label, drt=dr_technique, k=k, color_model=color_model)

    else:
        _, term_weight_name = metadata_latent_semantics(folder_path=folder_path, meta_data=label, drt=dr_technique, k=k, color_model=color_model)

    print("Open the file " + term_weight_name + " to see the output")











































