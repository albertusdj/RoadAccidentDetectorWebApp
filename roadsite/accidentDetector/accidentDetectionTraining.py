import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor

def readCSV(file):
    csv = pd.read_csv(file, sep="\t")
    data = pd.DataFrame({'tweets':csv['tweets'], 'isRoadIncident':csv['isRoadIncident']})[['tweets', 'isRoadIncident']]
    return data

def saveModel(model, file):
    output = open('models/%s' % file, 'wb')
    pkl.dump(model, output)
    output.close()

def loadModel(file):
    input = open('models/%s' % file, 'rb')


if __name__ == '__main__':
    preprocessor = Preprocessor()
    
    trainingFile = 'tweets/training-dataset.csv'
    trainingData = readCSV(trainingFile)

    X_train, X_test, y_train, y_test = train_test_split(trainingData['tweets'], trainingData['isRoadIncident'])
    
    pipeline = Pipeline([('bow',CountVectorizer(analyzer = preprocessor.text_process)),
                    ('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])
    pipeline.fit(X_train,y_train)

    # Validation
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy = ', accuracy*100, '%')

    saveModel(pipeline, 'multinomialNB.pkl')