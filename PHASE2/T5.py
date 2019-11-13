import pandas as pd
from T3 import metadata_latent_semantics
from phase1combined import main_fun
from sklearn import svm
import numpy as np
from scipy.stats import multivariate_normal
import random
import math


def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T.astype(np.float64))
    return mu, sigma

def multivariateGaussian(dataset,mu,sigma):
    try:
        p = multivariate_normal(mean=mu, cov=sigma)
        return p.pdf(dataset)
    except:
        return None


''' Method for Labelling an Un-labelled image '''
def image_classification(image_id, drt, k, color_model, option, folder_path):

    dics ={"left":"right", "right":"left", "male":"female", "female":"male",
           "dorsal":"palmar", "palmar":"dorsal", "with_accessories":"without_accessories",
           "without_accessories":"with_accessories"}


    option_file_path = folder_path + "_" + option + "_" + color_model + "_" + drt + "_" + str(k) + ".csv"


    ''' Creating the meta-data latent semantic for a given Specification'''
    if (drt == 'LDA' and color_model=='CM') or (drt == 'NMF' and color_model=='CM'):
        kmeans, option_comp, _ = metadata_latent_semantics(folder_path, option, drt, k, color_model)

    else:
        option_comp, _ = metadata_latent_semantics(folder_path, option, drt, k, color_model)
    _, test_matrix = main_fun(image_id, model=color_model)

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
    option_test = option_comp.transform(test_matrix)
    option_test = option_test.reshape(option_test.shape[1], )

    option_train_data = pd.read_csv(option_file_path).values

    option_train_data_image_ids = option_train_data[:,0]
    option_train_data_matrix = option_train_data[:,1:]

    ''' Applying 'one-class SVM' for predicting the label of unlabelled image_id'''

    print("Result obtained using One-Class SVM")
    oc_svm_clf = svm.OneClassSVM(gamma='auto', kernel='rbf', nu=0.02)
    oc_svm_clf.fit(option_train_data_matrix)
    pred = oc_svm_clf.predict(option_test.reshape(1, len(option_test)))
    if pred == 1:
        print(option)
    else:
        print(dics[option])

    ''' Applying 'Multi-variate Gaussian' for predicting the label of unlabelled image_id'''

    random.shuffle(option_train_data_matrix)
    mu, sigma = estimateGaussian(option_train_data_matrix)
    p = multivariateGaussian(option_train_data_matrix, mu, sigma)
    if p is not None:
        values = np.sort(p)[::-1]
        pred = multivariateGaussian(option_test, mu, sigma)
        threshold_index = int(0.95 * option_train_data_matrix.shape[0])
        threshold = values[threshold_index]
        print("Result obtained using Multi-variate Gaussian")
        if( threshold <= pred):
            print(option)
        else:
            print(dics[option])

    ''' Applying 'normal Gaussian' for predicting the label of unlabelled image_id  '''
    option_dists = []
    option_train_data_mean = np.mean(option_train_data_matrix, axis=0)
    option_train_data_std = np.std(option_train_data_matrix, dtype=np.float64, axis=0)
    for i in range(option_train_data_matrix.shape[0]):
        feature_test = option_train_data_matrix[i, :]
        val = 1
        for j in range(feature_test.shape[0]):
            val *= normpdf(feature_test[j], option_train_data_mean[j], option_train_data_std[j])
        option_dists.append(val)
    option_dists.sort(reverse=True)
    pred = 1
    for j in range(option_train_data_matrix.shape[1]):
        pred *= normpdf(option_test[j], option_train_data_mean[j], option_train_data_std[j])
    threshold_index = int(0.95 * option_train_data_matrix.shape[0])
    threshold = option_dists[threshold_index]
    print("Result obtained using normal Gaussian")
    if( threshold <= pred):
        print(option)
    else:
        print(dics[option])



if __name__ == '__main__':

    folder_path = input("please enter the folder path")

    color_model = input("Enter the model'CM' for Color Moments\n"
                        "'LBP' for Local Binary patterns:\n"
                        " 'HOG' for Histogram of Gradients:\n"
                        " 'SIFT' for Scale In-variant feature Transform ")

    k = int(input("Please give k value to give latent semantics: "))

    option = input("please enter the label")

    dr_technique = input("Enter the dimensionality reduction technqiue:\n"
                    "give PCA: Principle Component Analysis\n "
                    "SVD: Singular Value Decompostion\n "
                    "LDA: Latent Dirichlet Allocation\n"
                    "NMF: Non-Negative Matrix Factorization ")

    imageId = input("Please give the image-path: ")

    image_classification(image_id=imageId, drt=dr_technique, k=k, color_model=color_model, folder_path=folder_path, option= option)

    # image_classification(image_id='testing/Hand_0001395.jpg', drt='SVD', k=30, color_model='SIFT', folder_path='testing', option= 'without_accessories')
