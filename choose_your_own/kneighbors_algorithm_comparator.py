from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from time import time

n_neighbors_default = 5
weights_default = 'uniform'
algorithm_default = 'auto'
leaf_size_default = 30
p_default = 2
metric_default = 'minkowski'
metric_params_default = None
n_jobs_default = None


def calculate_best_accuracy_for_neighbors_range(features_train, labels_train, features_test, labels_test,
                                                neighbors_range,
                                                weights=weights_default, algorithm=algorithm_default,
                                                leaf_size=leaf_size_default,
                                                p=p_default, metric=metric_default, metric_params=metric_params_default,
                                                n_jobs=n_jobs_default,
                                                print_time=False, print_single_test_parameters=False,
                                                print_accuracy=False,
                                                print_parameters=False):
    max_accuracy = 0
    index_with_max_accuracy = 0

    executing_tests_time = time()
    for neighbors_number in neighbors_range:
        accuracy = calculate_accuracy_for_kneighbors_classifier(features_train, labels_train, features_test, labels_test,
                                                                neighbors_number, weights, algorithm, leaf_size,
                                                                p, metric, metric_params, n_jobs,
                                                                print_time, print_single_test_parameters, print_accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            index_with_max_accuracy = neighbors_number

    executing_tests_time = time() - executing_tests_time

    print_summary(algorithm, executing_tests_time, index_with_max_accuracy, leaf_size, max_accuracy, metric,
                  metric_params, n_jobs, neighbors_range, p, print_parameters, weights)
    return max_accuracy, index_with_max_accuracy


def calculate_accuracy_for_kneighbors_classifier(features_train, labels_train, features_test, labels_test,
                                                 n_neighbors=n_neighbors_default, weights=weights_default,
                                                 algorithm=algorithm_default, leaf_size=leaf_size_default,
                                                 p=p_default, metric=metric_default, metric_params=metric_params_default,
                                                 n_jobs=n_jobs_default,
                                                 print_time=False, print_parameters=False, print_accuracy=False):
    clf = create_kneighbors_classifier(algorithm, leaf_size, metric, metric_params, n_jobs, n_neighbors, p, weights)

    training_time = time()
    clf.fit(features_train, labels_train)
    if print_time:
        print("Time of training model: " + str(time() - training_time))

    predicted_labels = clf.predict(features_test)

    if print_parameters:
        parameters_string = "k neighbors classifier for parameters:\n"
        parameters_string += "n_neighbors: " + n_neighbors
        parameters_string += ", weights = " + str(weights)
        parameters_string += ", algorithm = " + str(algorithm)
        parameters_string += ", leaf_size = " + str(leaf_size)
        parameters_string += ", p = " + str(p)
        parameters_string += ", metric = " + str(metric)
        parameters_string += ", metric_params = " + str(metric_params)
        parameters_string += ", n_jobs = " + str(n_jobs)
        print parameters_string

    accuracy = accuracy_score(labels_test, predicted_labels)

    if print_accuracy:
        print("Accuracy of predicted labels: " + str(accuracy))
    return accuracy


def create_kneighbors_classifier(algorithm, leaf_size, metric, metric_params, n_jobs, n_neighbors, p, weights):
    if metric_params is not None:
        if n_jobs is not None:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                                       n_jobs=n_jobs)
        else:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params)
    else:
        if n_jobs is not None:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric, n_jobs=n_jobs)
        else:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                       leaf_size=leaf_size, p=p, metric=metric)
    return clf


def print_summary(algorithm, executing_tests_time, index_with_max_accuracy, leaf_size, max_accuracy, metric,
                  metric_params, n_jobs, neighbors_range, p, print_parameters, weights):
    if print_parameters:
        parameters_string = "Calculating for parameters:\n"
        parameters_string += "n_neighbors in range: " + str(neighbors_range[0]) + " - " + str(
            neighbors_range[len(neighbors_range) - 1])
        parameters_string += ", weights = " + str(weights)
        parameters_string += ", algorithm = " + str(algorithm)
        parameters_string += ", leaf_size = " + str(leaf_size)
        parameters_string += ", p = " + str(p)
        parameters_string += ", metric = " + str(metric)
        parameters_string += ", metric_params = " + str(metric_params)
        parameters_string += ", n_jobs = " + str(n_jobs)
        print(parameters_string)
        print("Executing tests time: " + str(executing_tests_time))
    print("Best accuracy: " + str(max_accuracy) + " for n_neighbors = " + str(index_with_max_accuracy))
