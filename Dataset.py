import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
  def __init__(self):
    pass

  #import dataset function and transform to numpy arrays by using values function
  def importDataset(dataObj):
      data = pd.read_csv(dataObj)
      data = data.values
      return data

  #import dataset function and transform to onedimensional numpy array
  def importLabels(dataObj):
      labels = pd.read_csv(dataObj)
      labels = np.array(labels).squeeze()
      return labels

  #Train_test_split function by sklearn to split loaded data into train and test data by scope of 80/20
  def splitTrainTestData(self, xval, yval):
      xtrain, xtest, ytrain, ytest = train_test_split(xval, yval, test_size=0.2, random_state=10)
      return xtrain, xtest, ytrain, ytest

  #Normalize data values to a range of 0 to 1 for better neural network performance
  def normalizeData(self, dataObj):
      normalizedData = dataObj / 378.0
      return normalizedData
