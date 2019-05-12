from os import listdir
from os.path import isfile, join
import re
import numpy as np

PATTERN = '_(.*)\.'

DIRECTORY = "data\\"


def read_directory(directory):
    only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    i = 0
    matrix = ["0", "a"]  # TODO remove it
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
    return matrix


# 1) Take reviews from data\neg and data\pos
# 2) Parse them and return np.stack with star and text
def take_train_data():
    pos = read_directory(DIRECTORY + "pos\\")
    neg = read_directory(DIRECTORY + "neg\\")
    return np.vstack((pos, neg))