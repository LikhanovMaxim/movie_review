import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
from numpy import load
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('wordnet')

REVIEW = "review"

FILE = "testData.tsv"
DATA_FOR_LEARNING = "labeledTrainData_simple.tsv"


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
        # >>> wnl = WordNetLemmatizer()
        # >>> print(wnl.lemmatize('cats'))
        # cat
    #
    # 6. Присоединим слова к одной строке, разделенной пробелом, и вернем результат.
    return " ".join(words)
    # return words


def normalization_text(raw_data):
    # Создание пустого списка и добавление чистых обзоров (отзывов) по одному
    number_of_reviews = len(raw_data[REVIEW])

    print("Cleaning and parsing the training set movie reviews... \n")
    net_initial_reviews = []
    for i in range(0, number_of_reviews):
        # Если индекс равномерно делится на 1000, распечатайте сообщение
        if (i + 1) % 1000 == 0:
            print("Review %d of %d\n" % (i + 1, number_of_reviews))
        cleaned = normalization(raw_data[REVIEW][i], True, True)
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


def testing(test):
    wnl = WordNetLemmatizer()
    # TODO check it. is it OK?
    words = ['got', 'watched', 'watches']
    words = [wnl.lemmatize(w) for w in words]
    print(words)
    # print(test.shape)
    # print(test)
    # print(test["sentiment"])
    # print(test["id"])
    # print(test["review"])
    # check = test[REVIEW][0]
    # print(check)
    # print(normalization(check))
    # print(normalization(check, True))
    # print("rev_of_ws")
    # # print(normalization(check))
    # print(normalization(check, True, True))


def bag_of_words(norm_text):
    # Инициализируем объект «CountVectorizer», который представляет собой
    # инструмент словаря scikit-learn.
    counter_of_vectors = TfidfVectorizer(analyzer="word",
                                         tokenizer=None,
                                         preprocessor=None,
                                         stop_words=None)
    # TODO Check: got an unexpected keyword argument 'max_func'
    composition_data_func = counter_of_vectors.fit_transform(norm_text)
    print(composition_data_func)
    # Numpy массивы легко работают, поэтому преобразуем получившийся результат в массив
    composition_data_func = composition_data_func.toarray()
    print("epta")
    print(composition_data_func)
    # Посмотрим на слова в словаре
    # TODO!!!!!!!!!!!!
    # vocab = composition_data_func.get_func_names()
    # print(vocab)
    # vocab
    # print("Training the random forest...")
    # forest = RandomForestClassifier(n_estimators=100)


def main():
    test = pd.read_csv(DATA_FOR_LEARNING, header=0, delimiter="\t", quoting=3)
    # testing(test)
    norm_text = normalization_text(test)
    print(norm_text)
    print(norm_text[0])
    bag_of_words(norm_text)


if __name__ == '__main__':
    main()
