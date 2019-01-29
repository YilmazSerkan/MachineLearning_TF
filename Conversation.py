from Dataset import Dataset
import pandas as pd
import numpy as np
import numpy

class Conversation(Dataset):

    #python constructor for conversation instances
    def __init__(self, dataPath):
        self.dataPath = dataPath

    #inherited import data function of class Dataset
    def importTextDataset(dataObj):
        convdata = pd.read_csv(dataObj.dataPath)
        convdata = convdata.values
        return convdata
