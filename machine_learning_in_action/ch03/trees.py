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
            reduced_dataset.extend(feature_vector[axis + 1:])
            splited_dataset.append(reduced_dataset)
    return splited_dataset


def choose_best_feature_to_split(dataset):
    """calculate each entropy of every feature, choose the best feature to split the dataset

    :param dataset: dataset waiting to be splited

    :return best_feature: best feature should be chosen to split the dataset
    """
    number_of_features = len(dataset[0]) - 1
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(number_of_features):
        feature_list = [example[i] for example in dataset]
        unique_value = set(feature_list)
        current_entropy = 0.0
        for each_value in unique_value:
            splited_dataset = split_dataset(dataset, i, each_value)
            prob = len(splited_dataset) / float(len(dataset))
            current_entropy += prob * calculate_shannon_entropy(splited_dataset)
        current_info_gain = base_entropy - current_entropy
        if (current_info_gain > best_info_gain):
            best_info_gain = current_info_gain
            best_feature = i
    return best_feature
