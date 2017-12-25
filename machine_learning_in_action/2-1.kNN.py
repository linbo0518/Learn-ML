import numpy as np
import operator


def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifiy0(input_X, dataset, label, k):
    """simplify kNN classifier

    :param input_X: one data to be classified 
    :param dataset: training dataset
    :param label: training label
    :param k: 

    :return sorted_label_count[0][0]: the most popular label
    """
    # Calculate the distance between the input_X and each example of the training set
    dataset_size = dataset.shape[0]
    minus_matrix = np.tile(input_X, (dataset_size, 1)) - dataset
    square_minus_matrix = minus_matrix**2
    square_distances = square_minus_matrix.sum(axis=1)
    distances = square_distances**0.5
    sorted_distance_indices = distances.argsort()
    # Choose the closest example of K
    label_count = {}
    for i in range(k):
        vote_label = label[sorted_distance_indices[i]]
        label_count[vote_label] = label_count.get(vote_label, 0) + 1
    # Sorted the labels
    sorted_label_count = sorted(
        label_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]
