import os
import operator
import numpy as np
from T3 import metadata_latent_semantics
import pandas as pd
import matplotlib.pyplot as plt
from phase1combined import main_fun, cosine_similarity
from sklearn.preprocessing import MinMaxScaler

''' Function to Visualize the top 'm' similar images'''
def plot_figures(figures, nrows = 1, ncols=1):

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)

    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

''' Method which gives the top 'm' similar images by loading the k-latent semantics of a (meta-data, color-model , drt, k) 
 file created in the task-3'''

def metadata_similarity_measure(folder_path, meta_data, test_image_path, drt, k, m, color_model):

    sub_filename = folder_path + "_" + meta_data + "_" + color_model + "_" + drt + "_" + str(k) + ".csv"

    ''' if no such file representing k-latent semantics of a (meta-data, color-model , drt, k)  triplet exists, 
     then the 'k_latent_semantics' method is called to create the file'''

    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM'):
        kmeans, comp, _ = metadata_latent_semantics(folder_path=folder_path,
                                  meta_data=meta_data,
                                  drt=drt,
                                  k=k,
                                  color_model=color_model)
    else:
        comp, _ = metadata_latent_semantics(folder_path=folder_path,
                                  meta_data=meta_data,
                                  drt=drt,
                                  k=k,
                                  color_model=color_model)

    data = pd.read_csv(sub_filename)
    data = data.values
    proj_mat = data[:,1:]
    image_ids = data[:,0]
    image_ids = image_ids.tolist()

    b = {}
    ''' Calculating the k-latent semantics for the query image'''
    _, test_matrix = main_fun(test_image_path, model=color_model)
    test_matrix = test_matrix.reshape(1, len(test_matrix))

    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM'):
        final_matrix = []
        for j in range(0, test_matrix.shape[1], 9):
            final_matrix.append(list(test_matrix[0, j:j+9]))
        final_matrix = np.array(final_matrix)
        test_labels = kmeans.predict(final_matrix)
        final_store_matrix = np.zeros((1, 400))
        for j in range(test_matrix.shape[1]//9):
            final_store_matrix[0, test_labels[j]] += 1
        test_matrix = final_store_matrix

    test_k = comp.transform(test_matrix)

    ''' Scaling the latent semantic data in range 0 to 1'''
    scaler = MinMaxScaler()
    proj_mat = scaler.fit_transform(proj_mat)
    test_k = scaler.transform(test_k)

    test_k = test_k.reshape(test_k.shape[1], )

    for i in range(len(image_ids)):
        b[image_ids[i]] = proj_mat[i, :]
    distances = {}
    i = 1

    ''' Calculating the Cosine Similarity between the query image and each training image of given meta-data
        in k-dimensional space'''
    for imageid, feature in b.items():
        distances[imageid] = cosine_similarity(test_k, feature)

    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1), reverse=True)

    similarity_ranking = {}
    figures = {}

    for tup in sorted_distances:
        figures['Similar-Image' + str(i) + ':' + tup[0]] = plt.imread('Hands/' + tup[0] + ".jpg")
        similarity_ranking[tup[0]] = tup[1]
        i += 1
        if i == m + 1:
            break
    return similarity_ranking, figures


if __name__ == '__main__':

    imagefolder_path = input('Please give the path to images folder: ')

    color_model = input("Enter the model'CM' for Color Moments\n"
                        "'LBP' for Local Binary patterns:\n"
                        " 'HOG' for Histogram of Gradients:\n"
                        " 'SIFT' for Scale In-variant feature Transform ")

    k =  int(input("Enter the value of k to find k-latent semantics: "))

    m = int(input("Enter the value of m to return m-similar images: "))

    label = input("Please give the label: ")

    drt = input("Enter the dimensionality reduction technqiue:\n"
                    "give PCA: Principle Component Analysis\n "
                    "SVD: Singular Value Decompostion\n "
                    "LDA: Latent Dirichlet Allocation\n"
                    "NMF: Non-Negative Matrix Factorization\n ")


    image_path = input("Please give query_image_path: ")


    similarity_ranking, figures = metadata_similarity_measure(folder_path=imagefolder_path,meta_data=label, test_image_path=image_path, drt=drt,k=k,m=m, color_model=color_model)

    plot_figures(figures,5,2)

    print(similarity_ranking)






