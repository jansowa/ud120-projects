#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# print "Cutting some data..."
# features_train = features_train[:len(features_train)/1000]
# labels_train = labels_train[:len(labels_train)/1000]

print "Creating classifier..."
classifier = SVC(kernel="rbf", C=10000.0)

print "Fitting data..."
fittingTime = time()
classifier.fit(features_train, labels_train)
print("Fitting time: " + str(time() - fittingTime))

print "Predicting labels..."
predicted_labels = classifier.predict(features_test)

print "Calculating accuracy"
accuracy = accuracy_score(labels_test, predicted_labels)
print("Accuracy = " + str(accuracy))

# print("Prediction for 10: " + str(predicted_labels[10]))
# print("Prediction for 26: " + str(predicted_labels[26]))
# print("Prediction for 50: " + str(predicted_labels[50]))
print("Number of Chris mails: " + str(sum(predicted_labels)))

#########################################################


