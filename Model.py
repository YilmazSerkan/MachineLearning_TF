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

    #Usage fit function by tf.keras for training
    def trainModel(self,model, xtrain, ytrain):
        #model = Model.buildModel()
        model.fit(xtrain, ytrain, epochs=5)

    #Usage of evaluate fuction by tf.keras for test
    def testModel(self,model,xtest,ytest):
        test_loss, test_acc = model.evaluate(xtest, ytest)
        print('Test accuracy:', test_acc)

    #Usage of predict fuction by tf.keras for prediction
    def executePrediction(self, model, convesationObj):
        predictions = model.predict(convesationObj)
        return predictions

"""Execution of two trainings & two tests for best model preformance by using different Datasets:
   Dataset DB.csv --> Represents documents tagged with specific keywords
   Dataset Usage.csv --> Represents keywords collected from past conversations
   Execution of a prediction by using a dataset which simulates a conversation"""
