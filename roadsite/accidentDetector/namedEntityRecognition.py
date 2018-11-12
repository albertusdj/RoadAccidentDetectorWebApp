
import nltk
import pandas as pd
import sklearn
import scipy.stats
import sklearn_crfsuite

from preprocessor import Preprocessor
from itertools import chain
from sklearn.metrics import make_scorer
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

def readCSV(file):
    csv = pd.read_csv(file, encoding = "ISO-8859-1")
    return csv

def extractFeatureFromWords(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'isSupper': word.isupper(),
        'isTitle': word.istitle(),
        'isDigit': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:lower': word1.lower(),
            '-1:isTitle': word1.istitle(),
            '-1:isSuper': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:lower': word1.lower(),
            '+1:isTitle': word1.istitle(),
            '+1:isSuper': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def extractFeatureFromSentences(sent):
    return [extractFeatureFromWords(sent, i) for i in range(len(sent))]

def labelsExtraction(sent):
    return [label for token, postag, label in sent]

def tokenize(sent):
    return [token for token, postag, label in sent]

def featureExtraction(data):
    # Even data splitting
    train_sentences = data[:10000]
    test_sentences = data[10000:20000]

    X_train = [extractFeatureFromSentences(sentence) for sentence in train_sentences]
    y_train = [labelsExtraction(sentence) for sentence in train_sentences]

    X_test = [extractFeatureFromSentences(sentence) for sentence in test_sentences]
    y_test = [labelsExtraction(sentence) for sentence in test_sentences]

    return (X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    preprocessor = Preprocessor()

    # Load data
    # Source: Kaggle
    # https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
    trainingFile = 'datasets/ner-dataset.csv'
    trainCSV = readCSV(trainingFile)

    sentences = preprocessor.processSentences(trainCSV.values)

    X_train, y_train, X_test, y_test = featureExtraction(sentences)

    # Training
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove('O')

    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, 
                          y_pred,
                          average='weighted', 
                          labels=['B-geo', 'I-geo']))
    




