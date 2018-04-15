import os
import numpy as np


wordlist_path = r'wordlist.txt'


def extract(text_array, words_to_find):
    x = []

    def contains(to_find):
        for t in text_array:
            if t == to_find:
                return 1
        return 0

    for word in words_to_find:
        x.append(contains(word))

    return x


def get_wordlist(path):
    return read_file(path).splitlines()


def read_file(path):
    with open(path, 'rb') as file:
        return file.read()


def create_data(path):
    x_data = []
    y_data = []

    labels = os.listdir(path)
    for index_of_label in range(len(labels)):
        file_directory = '{}\\{}'.format(path, labels[index_of_label])
        files = os.listdir(file_directory)
        for file in files:
            content = read_file('{}\\{}'.format(file_directory, file)).splitlines()
            x_data.append(extract(content, get_wordlist(wordlist_path)))
            y_data.append(index_of_label)

    return np.array(x_data), np.array(y_data), np.array(labels)
