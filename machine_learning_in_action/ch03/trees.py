from math import log


def create_dataset():
    """create a simple dataset for test

    :return dataset: created dataset
    :return label: created label
    """
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'],
               [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    return dataset, label


def calculate_shannon_entropy(dataset):
    """calculate the entropy of dataset

    :param dataset: input dataset
    """
    num_example = len(dataset)
    label_count = {}
    for feature_vector in dataset:
        current_label = feature_vector[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    shannon_entropy = 0.0
    for each_label in label_count:
        prob = float(label_count[each_label]) / num_example
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_dataset(dataset, axis, value):
    splited_dataset = []
    for feature_vector in dataset:
        if feature_vector[axis] == value:
            reduced_dataset = feature_vector[:axis]
            reduced_dataset.extend(feature_vector[axis+1:])
            splited_dataset.append(reduced_dataset)
    return splited_dataset
