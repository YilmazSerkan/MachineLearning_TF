import pandas as pd
import numpy as np
import numpy
import json
from Dataset import Dataset

class DictionaryManager:

    def __init__(self):
        pass

    #load json file with dictionary structure
    json_data = open(r"C:\Users\Serkan Yilmaz\PycharmProjects\MachineLearning_TF\dictkeywords.json").read()
    my_dict = json.loads(json_data)
    reverse_word_index = dict([(value, key) for (key, value) in my_dict.items()])

    #decode number reprensetation of keywords into initial strings by using dictionary
    def decode_keywords(self, dataObj):
        return ' '.join([DictionaryManager.reverse_word_index.get(i, '?') for i in dataObj])

    #encode string keywords into number reprensentation by using dictionary
    def encode_keywords(self, dataObj):
        newList = []
        for i in range(len(dataObj)):
            for j in range(len(dataObj[i])):
                mykeys = dataObj[i]
            newList.append([DictionaryManager.my_dict[k] for k in mykeys if k in DictionaryManager.my_dict])
        textToNumList = np.asarray(newList)
        return textToNumList
