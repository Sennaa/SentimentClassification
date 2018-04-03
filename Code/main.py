# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:15:23 2018

@author: Senna
"""

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_data(path, train, test):
    print("  Load data...")
    with open(path+train, encoding='latin1') as f:
        train_data      = np.loadtxt(f, dtype = np.str, delimiter = "\t")
    with open(path+test, encoding='latin1') as f:
        test_sentences  = np.loadtxt(f, dtype = np.str, delimiter = "\t")
    
    train_labels    = train_data[:,0]
    train_sentences = train_data[:,1]
    
    return train_sentences, train_labels, test_sentences

def create_tfidf(sentences):
    print("    Create tfidf...")
    # Set min_df to 0.01, to make sure redundant words are not included
    vectorizer = CountVectorizer(ngram_range=(1,2), stop_words='english', lowercase=True, min_df=0.01)
    tfidf = vectorizer.fit(sentences)
    return tfidf

def preprocess(tfidf, sentences):
    print("  Preprocess sentences")
    tfidf_transform = tfidf.transform(sentences)
    return tfidf_transform.toarray()

def evaluate(labels, predictions):
    print("    Evaluate model...")
    # Evaluate model with accuracy as metric
    accuracy    = list(predictions == labels).count(True) / len(predictions)
    #print("Accuracy: " + str(accuracy))
    return accuracy

def train_classifier(train, train_sentences, train_labels):
    print("  Train classifier...")
    # First, simple classifier
    clf = GaussianNB() # Accuracy: 0.97 (+/- 0.10) with min_df = 0.01
    # Then, try other classifier(s)
    #clf = RandomForestClassifier(n_estimators = 100, random_state = 95) # Accuracy: 0.94 (+/- 0.15) with default min_df
    clf.fit(train_sentences, train_labels)
    if train:
        scores = cross_val_score(clf, train_sentences, train_labels, cv=10)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return(clf)

def predict(clf, data):
    print("  Make predictions...")
    predictions = clf.predict(data)
    predictions = np.array(predictions)
    print(predictions)
    return(predictions)

def save_predictions(predictions, filepath = "../predictions/predictions.csv"):
    print("  Save predictions...")
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(filepath, header="Predictions", index=False)

def sentiment_classification(testsentence, train = False, path = "../Data/", trainfile = "traindata.txt", testfile = "testdata.txt"):
    print("Sentiment Classification")
    print(testsentence)
    testsentence    = [testsentence]
    train_sentences, train_labels, test_sentences = load_data(path, trainfile, testfile)
    if train:
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(train_sentences, train_labels, test_size=0.3, random_state=95)
    tfidf           = create_tfidf(train_sentences)
    train_tfidf     = preprocess(tfidf, train_sentences)
    clf             = train_classifier(train, train_tfidf, train_labels)
    if train:
        test_tfidf      = preprocess(tfidf, test_sentences)
        predictions     = predict(clf, test_tfidf)
        evaluate(test_labels, predictions)
    else:
        test_tfidf = preprocess(tfidf, testsentence)
        predict(clf, test_tfidf)
    #save_predictions(predictions)
    
if __name__ == '__main__':
    sentiment_classification(sys.argv[1])