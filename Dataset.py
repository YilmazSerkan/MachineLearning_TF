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
