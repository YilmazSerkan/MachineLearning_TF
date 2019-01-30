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

def main():

    """First Training & Test with DB.csv Dataset"""
    #Load DB.csv Train Data
    dbCsvPath = "/Users/alenagerlinskaja/PycharmProjects/Machine_Learning/DB.csv"
    dBcolnames = (['Keyword_1','Keyword_2','Keyword_3','Keyword_4','Keyword_5','Keyword_6','Keyword_7','Keyword_8','Keyword_9','Keyword_10'])
    dBdata = DocumentBase(dbCsvPath, dBcolnames)
    dBdata = dBdata.importDataset()

    #Load DB.csv Labels
    lbsDBcolnames = ['Doc_ID']
    dBLabels = DocumentBase(dbCsvPath, lbsDBcolnames)
    dBLabels = dBLabels.importLabels()

    #Split DB Dataset into Train & Test Data
    d = Dataset()
    xtrainDB, xtestDB, ytrainDB, ytestDB = d.splitTrainTestData(dBdata, dBLabels)

    #Normalize Data to values between 0 & 1
    xtrainDB = d.normalizeData(xtrainDB)
    xtestDB = d.normalizeData(xtestDB)
