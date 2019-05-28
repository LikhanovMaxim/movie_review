import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from sklearn.feature_selection import VarianceThreshold
# import prepare_train_data as prepare

# https://stackabuse.com/applying-filter-methods-in-python-for-feature-selection/
BAG_OF_WORDS_FULL = 'bag_of_words_full'
BAG_OF_WORDS_STARS_FULL = 'bag_of_words_stars_full'
BAG_OF_WORDS_VOCAB_FULL = 'bag_of_words_vocab_full'
BAG_OF_WORDS_FULL_AFTER_FS = "full_after_fs_bag_of_words"
BAG_OF_WORDS_STARS_FULL_AFTER_FS = "full_after_fs_bag_of_words_start"
BAG_OF_WORDS_VOCAB_FULL_AFTER_FS = "full_after_fs_bag_of_words_vocab"
BAG_OF_WORDS_SMALL = 'bag_of_words_small'
BAG_OF_WORDS_STARS_SMALL = 'bag_of_words_stars_small'
BAG_OF_WORDS_VOCAB_SMALL = 'bag_of_words_vocab_small'


def prepare_test_dataset():
    standard_data = pd.read_csv("E:/git projects/datasets/train.csv",
                                nrows=40000)
    print(standard_data.shape)
    train_features, test_features, train_labels, test_labels = train_test_split(
        standard_data.drop(labels=['TARGET'], axis=1),
        standard_data['TARGET'],
        test_size=0.2,
        random_state=41)
    print("After split")
    print(train_features.shape)
    print(train_labels.shape)
    print(test_features.shape)
    print(test_labels.shape)
    return test_features, train_features


def removing_correlated_features():
    paribas_data = pd.read_csv("E:/git projects/datasets/train.csv", nrows=20000)
    print(paribas_data.shape)
    num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = list(paribas_data.select_dtypes(include=num_colums).columns)
    paribas_data = paribas_data[numerical_columns]
    print(paribas_data.shape)
    train_features, test_features, train_labels, test_labels = train_test_split(
        paribas_data.drop(labels=['TARGET', 'ID'], axis=1),
        paribas_data['TARGET'],
        test_size=0.2,
        random_state=41)
    correlated_features = set()
    correlation_matrix = paribas_data.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print(len(correlated_features))
    print(correlated_features)
    train_features.drop(labels=correlated_features, axis=1, inplace=True)
    test_features.drop(labels=correlated_features, axis=1, inplace=True)
    print(train_features.shape)
    print(test_features.shape)


def split_data():
    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_STARS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL,
    #                                                  BAG_OF_WORDS_STARS_FULL,
    #                                                  BAG_OF_WORDS_VOCAB_FULL)
    matrix, stars, vocab = take_bag_of_words(BAG_OF_WORDS_SMALL,
                                             BAG_OF_WORDS_STARS_SMALL,
                                             BAG_OF_WORDS_VOCAB_SMALL)
    print(matrix.shape)
    # paribas_data = matrix
    paribas_data = pd.DataFrame(matrix)
    stars = pd.DataFrame(stars)
    print(stars.shape)
    frames = [stars, paribas_data]
    res = pd.concat(frames, axis=1, sort=False)
    print('Concat:')
    print(res.shape)
    # TODO what is label in stars? Change them to TARGET or rename TARGET
    train_features, test_features, train_labels, test_labels = train_test_split(
        res.drop(labels=[0], axis=1),
        res[0],
        test_size=0.2,
        random_state=41)
    print(train_features.shape)
    print(train_labels.shape)
    print(train_features[0])
    print(train_labels[0])


NPY = ".npy"


def take_bag_of_words(file_matrix=BAG_OF_WORDS_FULL, file_stars=BAG_OF_WORDS_STARS_FULL,
                      file_vocab=BAG_OF_WORDS_VOCAB_FULL):
    matrix = np.load(file_matrix + NPY)
    stars = np.load(file_stars + NPY)
    vocab = np.load(file_vocab + NPY)
    # matrix = np.load(BAG_OF_WORDS_SMALL + NPY)
    # stars = np.load(BAG_OF_WORDS_STARS_SMALL + NPY)
    # vocab = np.load(BAG_OF_WORDS_VOCAB_SMALL + NPY)
    return matrix, stars, vocab


if __name__ == '__main__':
    split_data()
