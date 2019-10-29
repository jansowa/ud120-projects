#!/usr/bin/python

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

## the training data (features_train, labels_train) have both "fast" and "slow"
## points mixed together--separate them so we can give them different colors
## in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
###############################################################################

from kneighbors_algorithm_comparator import calculate_best_accuracy_for_neighbors_range
from adaboost_algorithm_comparator import calculate_best_accuracy_for_adaboost_classifier
from random_forest_comparator import calculate_best_accuracy_for_random_forest_classifier

calculate_best_accuracy_for_neighbors_range(features_train, labels_train, features_test, labels_test,
                                            weights='uniform', print_parameters=True)
print
calculate_best_accuracy_for_adaboost_classifier(features_train, labels_train, features_test, labels_test, algorithm="SAMME", print_parameters=True)
print
calculate_best_accuracy_for_random_forest_classifier(features_train, labels_train, features_test, labels_test, n_estimators_range=range(1, 10), max_feature=1)