from os import listdir
from os.path import isfile, join
import re
import numpy as np

PATTERN = '_(.*)\.'

DIRECTORY = "data\\"


def read_directory(directory, is_small):
    only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    i = 0
    matrix = ["7", "good"]  # TODO remove it. Bag of words has 0 & "".
    print("start handling files")
    for file in only_files:
        f = open(directory + file, "r", encoding="utf-8")
        text = f.read()
        star = re.search(PATTERN, file).group(1)
        row = [star, text]
        matrix = np.vstack((matrix, row))
        i = i + 1
        if (i + 1) % 100 == 0:
            print("Files %d" % (i + 1))
            if is_small:
                break
    return matrix


# 1) Take reviews from data\neg and data\pos
# 2) Parse them and return np.stack with star and text
def take_train_data(is_small):
    pos = read_directory(DIRECTORY + "pos\\", is_small)
    neg = read_directory(DIRECTORY + "neg\\", is_small)
    return np.vstack((pos, neg))
