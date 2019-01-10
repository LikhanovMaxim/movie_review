import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from numpy import load
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from os import listdir
from os.path import isfile, join

NPY = ".npy"

BAG_OF_WORDS_SMALL = 'bag_of_words_small'
BAG_OF_WORDS_STARS_SMALL = 'bag_of_words_stars_small'
BAG_OF_WORDS_VOCAB_SMALL = 'bag_of_words_vocab_small'
BAG_OF_WORDS_FULL = 'bag_of_words_full'
BAG_OF_WORDS_STARS_FULL = 'bag_of_words_stars_full'
BAG_OF_WORDS_VOCAB_FULL = 'bag_of_words_vocab_full'

BAG_OF_WORDS = 'labeledBow.feat'

PATTERN = '_(.*)\.'

DIRECTORY = "data\\"

MAX_FEATURES = 10000

nltk.download('stopwords')
nltk.download('wordnet')

REVIEW = 1

FILE = "testData.tsv"
DATA_FOR_LEARNING = "labeledTrainData_simple.tsv"
DATA_FOR_LEARNING_FULL = "labeledTrainData.tsv"


def normalization(blank_review, remove_stop_words=False, need_to_lemm=False):
    # Функция преобразования документа в список слов
    # Необязательно удалять все, кроме словарных слов.
    #
    # 1. Удаляем html
    text_review = BeautifulSoup(blank_review, "lxml").get_text()
    #
    # 2.  Удаляем символы
    text_review = re.sub("[^a-zA-Z]", " ", text_review)
    #
    # 3.  Преобразование слов в нижний регистр и их разделение
    words = text_review.lower().split()

    #
    # 5.  При необходимости удалим стоп-слова
    if remove_stop_words:
        # 4.   В Python поиск в наборе намного быстрее, чем поиск в списке,
        # поэтому преобразунм стоп-слова в набор
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    if need_to_lemm:
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(w) for w in words]
    #
    # 6. Присоединим слова к одной строке, разделенной пробелом, и вернем результат.
    return " ".join(words)


def normalization_text(raw_data):
    # Создание пустого списка и добавление чистых обзоров (отзывов) по одному
    number_of_reviews = len(raw_data[REVIEW])

    print("Cleaning and parsing the training set movie reviews... \n")
    net_initial_reviews = []
    for i in range(0, number_of_reviews):
        if (i + 1) % 1000 == 0:
            print("Review %d of %d\n" % (i + 1, number_of_reviews))
        cleaned = normalization(raw_data[REVIEW][i], True, True)
        net_initial_reviews.append(cleaned)
    return net_initial_reviews


def normalization_text_matrix(raw_data):
    # Создание пустого списка и добавление чистых обзоров (отзывов) по одному
    number_of_reviews = len(raw_data)

    print("Cleaning and parsing the training set movie reviews... \n")
    print(number_of_reviews)
    net_initial_reviews = []
    for i in range(0, number_of_reviews):
        # Если индекс равномерно делится на 1000, распечатайте сообщение
        if (i + 1) % 1000 == 0:
            print("Review %d of %d\n" % (i + 1, number_of_reviews))
        cleaned = normalization(raw_data[i][REVIEW], True, True)
        # print("cleaned: " + cleaned)
        net_initial_reviews.append(cleaned)
    return net_initial_reviews


def review_of_sent(blank_review, tokenizer, remove_stop_words=False, need_to_lemm=False):
    # Функция для разделения обзора(отзыва) на разобранные предложения.
    # Возвращает список предложений, где каждое предложение представляет собой список слов
    #
    # Используем токенизатор NLTK для разделения абзаца на предложения
    original_sents = tokenizer.tokenize(blank_review.decode('utf8').strip())
    #
    final_sents = []
    for raw_sent in original_sents:
        # Если предложение пустое, пропустим его
        if len(raw_sent) > 0:
            final_sents.append(normalization(raw_sent, remove_stop_words, need_to_lemm))

    # Возвращает список предложений (каждое предложение представляет собой список слов)
    return final_sents


def unzip():
    data = load('imdb.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


def write_to_file(train_data, file_name):
    np.save(file_name, train_data)


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


def print_often_words(train_data, vocab, length=10):
    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data, axis=0)
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True)[0:length]:
        print(count, tag)


def tfidf_method(norm_text):
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=MAX_FEATURES)
    matrix = vectorizer.fit_transform(norm_text)
    return matrix.toarray(), vectorizer.get_feature_names()


def print_info_matrix(matrix, vocab):
    print(matrix.shape)
    print(vocab[0:100])
    print_often_words(matrix, vocab)


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


def take_train_data():
    pos = read_directory(DIRECTORY + "pos\\")
    neg = read_directory(DIRECTORY + "neg\\")
    return np.vstack((pos, neg))


def column(matrix, i):
    return [row[i] for row in matrix]


def prepare_data():
    train_data = take_train_data()

    norm_text = normalization_text_matrix(train_data)

    print("\nBag of words")
    [matrix, vocab] = bag_of_words(norm_text)

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
