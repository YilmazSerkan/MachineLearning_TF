import tensorflow as tf
from tensorflow import keras
from Usage import Usage
from Dataset import Dataset
from Dictionary import DictionaryManager
from Conversation import Conversation
from DocumentBase import DocumentBase

class Model():

    def __init__(self):
        pass

    #Usage of keras.Sequential & compile function by tensorflow keras
    def buildModel(self):
        model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(7, activation=tf.nn.softmax)])
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
