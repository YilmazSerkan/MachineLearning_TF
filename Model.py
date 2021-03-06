import tensorflow as tf
from tensorflow import keras
from Usage import Usage
from Dataset import Dataset
from Dictionary import DictionaryManager
from Conversation import Conversation
from DocumentBase import DocumentBase
import numpy as np


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
    dbCsvPath = "/Users/alenagerlinskaja/PycharmProjects/MachineLearning_TF/DB.csv"
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

    #Load Usage.csv Train Data
    usageCsvPath = "/Users/alenagerlinskaja/PycharmProjects/MachineLearning_TF/Usage.csv"
    usagecolnames = (['Keyword_1','Keyword_2','Keyword_3','Keyword_4','Keyword_5','Keyword_6','Keyword_7','Keyword_8','Keyword_9','Keyword_10'])
    usageData = Usage(usageCsvPath, usagecolnames)
    usageData = usageData.importDataset()
    #print usageData

    #Load Usage.csv Labels
    lbsusagecolnames = ['Doc_ID']
    usageLabels = Usage(usageCsvPath,lbsusagecolnames)
    usageLabels = usageLabels.importLabels()

    #Split Usage Dataset into Train & Test Data
    d = Dataset()
    xtrainUsage, xtestUsage, ytrainUsage, ytestUsage = d.splitTrainTestData(usageData, usageLabels)

    # Normalize usage data to values between 0 & 1
    xtrainUsage = d.normalizeData(xtrainUsage)
    xtestUsage = d.normalizeData(xtestUsage)

    #Load conversation keywords(strings) for prediction
    conversation = Conversation("/Users/alenagerlinskaja/PycharmProjects/MachineLearning_TF/conversation.csv")
    conversationStrData = conversation.importTextDataset()

    #Encode string values to numbers by using a dictionary
    dict = DictionaryManager()
    conversationNumData= dict.encode_keywords(conversationStrData)
    convData = conversation.normalizeData(conversationNumData)


    """First Training & Test with DB.csv Dataset"""
    #Build & compile the model
    m = Model()
    model = m.buildModel()

    print '==========TRAINING=========='
    #Start Training by using train data & train lables from DB.csv
    m.trainModel(model,xtrainDB,ytrainDB)

    print '==========TEST=========='
    #Start Test/Evaluation by using test data & test lables from DB.csv
    m.testModel(model, xtestDB,ytestDB)

    print '==========TRAINING=========='
    """Second Training & Test with Usage.csv Dataset"""
    #Start Training by using train data & train lables from Usage.csv
    m.trainModel(model, xtrainUsage, ytrainUsage)

    print '==========TEST=========='
    #Start Test/Evaluation by using test data & test lables from Usage.csv
    m.testModel(model, xtestUsage, ytestUsage)

    """Execute a prediction for Conversation Data"""
    #Execute a prediction
    predictions = m.executePrediction(model, convData)

    print '==========PREDICTIONS=========='

    for i in range(len(conversationStrData)):
        print '------------- Prediction', i,'--------------'
        print conversationStrData[i]
        print 'PREDICTION LABEL', np.argmax(predictions[i])

main()
