#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
print("Creating classifier...")
classifier = GaussianNB()
print("Fitting data...")
fittingTime = time()
classifier.fit(features_train, labels_train)
print("Fitting time: " + str(round(time() - fittingTime, 3)) + "s")
print("Predicting labels...")
predictingTime = time()
predictedLabels = classifier.predict(features_test)
print("Predicting time: " + str(round(time() - predictingTime, 3)) + "s")
print("Calculating accuracy...")
calculatingAccuracyTime = time()
accuracy = accuracy_score(labels_test, predictedLabels)
print("Calculating accuracy time: " + str(time() - calculatingAccuracyTime) + "s")
print("accuracy = " + str(accuracy))
#########################################################


