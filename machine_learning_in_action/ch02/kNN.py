import os
import operator
import numpy as np


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


def file2matrix(filename):
    """split input file to example matrix and label vector

    :param filename: input file

    :return example_matrix: data matrix
    :return label_vector: label vector
    """
    # Get file lines
    file_obj = open(filename)
    lines_list = file_obj.readlines()
    file_obj.close()
    number_of_lines = len(lines_list)
    example_matrix = np.zeros((number_of_lines, 3))
    label_vector = []
    index = 0
    for each_line in lines_list:
        each_line = each_line.strip()
        list_from_line = each_line.split('\t')
        example_matrix[index, :] = list_from_line[0:3]
        label_vector.append(list_from_line[-1])
        index += 1
    return example_matrix, label_vector


def auto_normalization(dataset):
    """make dataset normalization, range from 0 to 1

    :param dataset: input dataset

    :return normalized_dataset: normalized dataset
    :return value_range: range of value
    :return min_value: numerical minimum data
    """
    min_value = dataset.min(0)
    max_value = dataset.max(0)
    value_range = max_value - min_value
    normalized_dataset = np.zeros(np.shape(dataset))
    ndimension_training = dataset.shape[0]
    normalized_dataset = dataset - np.tile(min_value, (ndimension_training, 1))
    normalized_dataset = normalized_dataset / np.tile(value_range,
                                                      (ndimension_training, 1))
    return normalized_dataset, value_range, min_value


def image2vector(filename):
    """convert image to vector

    :param filename: input image file, 32 * 32 pixels

    :return image_vector: 1 * 1024 vector
    """
    image_vector = np.zeros((1, 1024))
    file_obj = open(filename)
    for i in range(32):
        each_line = file_obj.readline()
        for j in range(32):
            image_vector[0, 32 * i + j] = int(each_line[j])
    file_obj.close()
    return image_vector


def handwriting_classifiy():
    """handwriting digits classifier
    """
    # get trainingDigits directory content
    handwriting_label = []
    training_file_list = os.listdir('trainingDigits')
    ndimension_training = len(training_file_list)
    training_matrix = np.zeros((ndimension_training, 1024))
    # get label name from training file name
    for i in range(ndimension_training):
        file_name = training_file_list[i]
        file_name_noext = file_name.split('.')[0]
        label_name = int(file_name_noext.split('_')[0])
        handwriting_label.append(label_name)
        training_matrix[i, :] = image2vector('trainingDigits/%s' % file_name)
    # get testDigits directory content
    test_file_list = os.listdir('testDigits')
    ndimension_test = len(test_file_list)
    # get label name from test file name
    error_count = 0.0
    for i in range(ndimension_test):
        file_name = test_file_list[i]
        file_name_noext = file_name.split('.')[0]
        label_name = int(file_name_noext.split('_')[0])
        test_vector = image2vector('testDigits/%s' % file_name)

        classifiy_result = classifiy0(test_vector, training_matrix,
                                      handwriting_label, 3)
        print('the classifier come back with: %d, the real number is %d' %
              (classifiy_result, label_name))
        if classifiy_result != label_name:
            error_count += 1.0
    print('the total number of error is: %d' % error_count)
    print('the total error rate is: %f' % (error_count / ndimension_test))
