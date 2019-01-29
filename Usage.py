from Dataset import Dataset
import pandas as pd
import numpy as np
import numpy

class Usage(Dataset):

    # python constructor specifies usage instance should consist out of datapath & column names
    def __init__(self, dataPath, colnames):
        self.dataPath = dataPath
        self.colnames = colnames

    #inherited import dataset function of class Dataset
    def importDataset(dataObj):
        train_data = pd.read_csv(dataObj.dataPath, usecols = dataObj.colnames)
        train_data = numpy.nan_to_num(train_data)
        train_data = train_data.astype(int)
        #print train_data
        return train_data
