#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

number_of_train_features = len(features_train[0])
print("There are " + str(number_of_train_features) + " features in train data.")

print("Creating classifier...")
classifier = DecisionTreeClassifier(min_samples_split=40)

print("Training model...")
fittingTime = time()
classifier.fit(features_train, labels_train)
print("Time of fitting data: " + str(time() - fittingTime))

print("Predicting labels...")
predicted_labels = classifier.predict(features_test)

print("Calculating accuracy")
accuracy = accuracy_score(labels_test, predicted_labels)

print("Accuracy of predicted labels is: " + str(accuracy))

#########################################################
### your code goes here ###


#########################################################


