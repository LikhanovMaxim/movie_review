import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import prepare_train_data as prepare

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


def remove_constant_features():
    test_features, train_features = prepare_test_dataset()

    print("Start delete")
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train_features)
    # print_info_non_consts(constant_filter, train_features)
    # print_constants(constant_filter, train_features)
    train_features = constant_filter.transform(train_features)
    test_features = constant_filter.transform(test_features)

    print("In total:")
    print(train_features.shape)
    print(test_features.shape)


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


def print_info_non_consts(constant_filter, train_features):
    length_non_constant = len(train_features.columns[constant_filter.get_support()])
    print(length_non_constant)


def print_constants(constant_filter, train_features):
    constant_columns = [column for column in train_features.columns
                        if column not in train_features.columns[constant_filter.get_support()]]
    print(len(constant_columns))
    for column in constant_columns:
        print(column)


def customize_remove_consts():
    matrix, stars, vocab = prepare.take_tf_idf()
    print(matrix.shape)
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(matrix)
    # print_info_non_consts(constant_filter, train_features)
    # print_constants(constant_filter, train_features)
    train_features = constant_filter.transform(matrix)
    print("In total:")
    print(train_features.shape)


def remove_quasi_constant():
    test_features, train_features = prepare_test_dataset()
    # Remove consts
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train_features)

    len(train_features.columns[constant_filter.get_support()])

    constant_columns = [column for column in train_features.columns
                        if column not in train_features.columns[constant_filter.get_support()]]

    train_features.drop(labels=constant_columns, axis=1, inplace=True)
    test_features.drop(labels=constant_columns, axis=1, inplace=True)
    # Remove quasi_constant
    print("Remove quasi_constant")
    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(train_features)
    print(len(train_features.columns[qconstant_filter.get_support()]))
    qconstant_columns = [column for column in train_features.columns
                         if column not in train_features.columns[qconstant_filter.get_support()]]

    print(len(qconstant_columns))
    train_features = qconstant_filter.transform(train_features)
    test_features = qconstant_filter.transform(test_features)

    print(train_features.shape)
    print(test_features.shape)


def customize_quasi_constant():
    matrix, stars, vocab = prepare.take_tf_idf()
    print(matrix.shape)
    train_features = matrix

    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train_features)

    # len(train_features.columns[constant_filter.get_support()])

    constant_columns = [column for column in train_features.columns
                        if column not in train_features.columns[constant_filter.get_support()]]

    train_features.drop(labels=constant_columns, axis=1, inplace=True)
    print("Remove quasi_constant")

    qconstant_filter = VarianceThreshold(threshold=0.01)
    qconstant_filter.fit(train_features)
    print(len(train_features.columns[qconstant_filter.get_support()]))
    qconstant_columns = [column for column in train_features.columns
                         if column not in train_features.columns[qconstant_filter.get_support()]]

    print(len(qconstant_columns))
    train_features = qconstant_filter.transform(train_features)

    print(train_features.shape)


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


def customize_removing_correlated_features():
    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_STARS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL_AFTER_FS,
                                                     BAG_OF_WORDS_STARS_FULL_AFTER_FS,
                                                     BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_SMALL,
    #                                                  BAG_OF_WORDS_STARS_SMALL,
    #                                                  BAG_OF_WORDS_VOCAB_SMALL)
    print(matrix.shape)
    # paribas_data = matrix
    paribas_data = pd.DataFrame(matrix)
    # paribas_data = pd.read_csv("E:/git projects/datasets/train.csv", nrows=20000)
    # first run delete 601
    # paribas_data = paribas_data.head(1000)
    # paribas_data = paribas_data.iloc[:, : 10000]
    # second run delete 763 (25002, 65080)
    # paribas_data = paribas_data.iloc[:2000, 10000: 20000]
    # Third run delete 720 (25002, 64360)
    # paribas_data = paribas_data.iloc[:2000, 20000: 30000]
    # 4 run delete it takes so long
    # paribas_data = paribas_data.iloc[:2000, : 20000]
    paribas_data = paribas_data.iloc[:2000, 30000: 40000]
    print(paribas_data.shape)
    # paribas_data = smth(paribas_data)
    print(paribas_data.shape)
    # train_features, test_features, train_labels, test_labels = train_test_split(
    #     paribas_data.drop(labels=['TARGET', 'ID'], axis=1),
    #     paribas_data['TARGET'],
    #     test_size=0.2,
    #     random_ state=41)
    correlated_features = find_corr_features(paribas_data)
    # print(correlated_features)
    new_matrix = pd.DataFrame(matrix)
    new_matrix.drop(labels=correlated_features, axis=1, inplace=True)
    # test_features.drop(labels=correlated_features, axis=1, inplace=True)
    print(new_matrix.shape)
    prepare.write_to_file(new_matrix, BAG_OF_WORDS_FULL_AFTER_FS)
    prepare.write_to_file(stars, BAG_OF_WORDS_STARS_FULL_AFTER_FS)
    prepare.write_to_file(vocab, BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    print("Finish")
    # print(test_features.shape)


def find_corr_features(paribas_data):
    correlated_features = set()
    print("Before corr")
    correlation_matrix = paribas_data.corr()
    print("Find features which ")
    print(correlation_matrix.shape)
    print(correlation_matrix)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)
    print(len(correlated_features))
    return correlated_features


def smth(paribas_data):
    num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numerical_columns = list(paribas_data.select_dtypes(include=num_colums).columns)
    paribas_data = paribas_data[numerical_columns]
    return paribas_data


if __name__ == '__main__':
    # remove_constant_features()
    # customize_remove_consts()
    # remove_quasi_constant()
    # customize_quasi_constant()
    # removing_correlated_features()
    customize_removing_correlated_features()  # it works
