# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 14:15:23 2018

@author: Senna
"""

import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

def load_data(path, train, test):
    print("  Load data...")
    train_data      = np.loadtxt(path + train, dtype = np.str, delimiter = "\t", encoding = "utf8")
    test_sentences  = np.loadtxt(path + test, dtype = np.str, delimiter = "\t", encoding = "utf8")
    
    train_labels    = train_data[:,0]
    train_sentences = train_data[:,1]
    
    return(train_sentences, train_labels, test_sentences)

def create_tfidf(sentences):
    print("    Create tfidf...")
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', lowercase=True)
    tfidf = vectorizer.fit(sentences)
    return(tfidf)

def preprocess(tfidf, sentences):
    print("  Preprocess sentences")
    tfidf_transform = tfidf.transform(sentences)
    return(tfidf_transform.toarray())

def evaluate(labels, predictions):
    print("    Evaluate model...")
    # Evaluate model with accuracy as metric
    accuracy    = list(predictions == labels).count(True) / len(predictions)
    return(accuracy)

def train_classifier(train, train_sentences, train_labels):
    print("  Train classifier...")
    if (train):
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(train_sentences, train_labels, test_size = 0.2, random_state = 95)
    # First, simple classifier
    clf = GaussianNB()
    clf.fit(train_sentences, train_labels)
    if (train):
        predictions = predict(clf,test_sentences)
        print("      Accuracy: " + str(evaluate(test_labels, predictions)))
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
    tfidf           = create_tfidf(train_sentences)
    train_tfidf     = preprocess(tfidf, train_sentences)
    clf             = train_classifier(train, train_tfidf, train_labels)
    if (not train):
        test_tfidf  = preprocess(tfidf, testsentence)
        predictions = predict(clf, test_tfidf)
        #save_predictions(predictions)
    
if __name__ == '__main__':
    sentiment_classification(sys.argv[1])