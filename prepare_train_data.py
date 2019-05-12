import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import parser_data as parser
import normalization as norm

NPY = ".npy"

BAG_OF_WORDS_SMALL = 'bag_of_words_small'
BAG_OF_WORDS_STARS_SMALL = 'bag_of_words_stars_small'
BAG_OF_WORDS_VOCAB_SMALL = 'bag_of_words_vocab_small'

BAG_OF_WORDS_FULL = 'bag_of_words_full'
BAG_OF_WORDS_STARS_FULL = 'bag_of_words_stars_full'
BAG_OF_WORDS_VOCAB_FULL = 'bag_of_words_vocab_full'

# BAG_OF_WORDS = 'labeledBow.feat'

MAX_FEATURES = 10000


#
# FILE = "testData.tsv"
# DATA_FOR_LEARNING = "labeledTrainData_simple.tsv"
# DATA_FOR_LEARNING_FULL = "labeledTrainData.tsv"


def bag_of_words(norm_text):
    # Инициализируем объект «CountVectorizer», который представляет собой
    # инструмент словаря scikit-learn.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None)
    train_data = vectorizer.fit_transform(norm_text)
    train_data = train_data.toarray()
    vocab = vectorizer.get_feature_names()
    return train_data, vocab


# TODO it doesn't work / test
def tfidf_method(norm_text):
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=MAX_FEATURES)
    matrix = vectorizer.fit_transform(norm_text)
    return matrix.toarray(), vectorizer.get_feature_names()


def print_often_words(train_data, vocab, length=10):
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data, axis=0)
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True)[0:length]:
        print(count, tag)


def print_info_matrix(matrix, vocab):
    print(matrix.shape)
    print(vocab[0:100])
    print_often_words(matrix, vocab)


def column(matrix, i):
    return [row[i] for row in matrix]


def write_to_file(train_data, file_name):
    np.save(file_name, train_data)


def prepare_data_by_bag_of_words(is_small=True, is_write=False):
    train_data = parser. take_train_data(is_small)

    norm_text = norm.normalization_text_matrix(train_data)

    print("\nBag of words")
    [matrix, vocab] = bag_of_words(norm_text)

    print_info_matrix(matrix, vocab)
    if is_write:
        write_to_file(matrix, BAG_OF_WORDS_SMALL)
        write_to_file(column(train_data, 0), BAG_OF_WORDS_STARS_SMALL)
        write_to_file(vocab, BAG_OF_WORDS_VOCAB_SMALL)


def take_bag_of_words():
    matrix = np.load(BAG_OF_WORDS_FULL + NPY)
    stars = np.load(BAG_OF_WORDS_STARS_FULL + NPY)
    vocab = np.load(BAG_OF_WORDS_VOCAB_FULL + NPY)
    # matrix = np.load(BAG_OF_WORDS_SMALL + NPY)
    # stars = np.load(BAG_OF_WORDS_STARS_SMALL + NPY)
    # vocab = np.load(BAG_OF_WORDS_VOCAB_SMALL + NPY)
    return matrix, stars, vocab
