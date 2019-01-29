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
