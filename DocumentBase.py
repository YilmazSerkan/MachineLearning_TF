from Dataset import Dataset
import pandas as pd
import numpy as np
import numpy
from sklearn.model_selection import train_test_split

class DocumentBase(Dataset):

    #python constructor specifies that dataset instance should consist out of datapath & column names
    def __init__(self, dataPath, colnames):
        self.dataPath = dataPath
        self.colnames = colnames

    #inherited import dataset function of class Dataset
    def importDataset(dataObj):
        train_data = pd.read_csv(dataObj.dataPath, usecols = dataObj.colnames)
        train_data = numpy.nan_to_num(train_data)
        train_data = train_data.astype(int)
        return train_data

    #inherited import labels function of class Dataset
    def importLabels(dataObj):
        train_labels = pd.read_csv(dataObj.dataPath, usecols = dataObj.colnames)
        train_labels = np.array(train_labels).squeeze()
        return train_labels

    #inherited import split function of class Dataset - sklearn train test split function
    #splits dataset into train and test data in the scope of 80/20
    def splitTrainTestData(self, xval, yval):
        xtrain, xtest, ytrain, ytest = train_test_split(xval, yval, test_size=0.2, random_state=10)
        return xtrain, xtest, ytrain, ytest
