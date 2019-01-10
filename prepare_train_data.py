import pandas as pd
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
# REVIEW = "review"

FILE = "testData.tsv"
DATA_FOR_LEARNING = "labeledTrainData_simple.tsv"
DATA_FOR_LEARNING_FULL = "labeledTrainData.tsv"


def normalization(blank_review, remove_stop_words=False, need_to_lemm=False):
    # print(blank_review)
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


# It is not used
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


def write_to_file(train_data, file_name):
    # print(np.matrix(train_data))
    np.save(file_name, train_data)
    # output = pd.DataFrame(data={"id": train_data[0], "sentiment": train_data[1]})
    # df.to_csv(file_name, sep='\t')
    # # Скопируем результаты в таблицу данных pandas с колонкой «id» и
    # # столбец «настроение»
    # output = pd.DataFrame(data={"id": train_data["id"], "sentiment": result})
    # # Используем pandas для записи выходного файла, разделенного запятыми
    # output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)


def bag_of_words(norm_text):
    # Инициализируем объект «CountVectorizer», который представляет собой
    # инструмент словаря scikit-learn.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None)
    # TODO is it needed?
    # max_features=MAX_FEATURES)
    train_data = vectorizer.fit_transform(norm_text)
    # print(train_data)
    # Numpy массивы легко работают, поэтому преобразуем получившийся результат в массив
    train_data = train_data.toarray()
    # print("toarray")
    # print(train_data.shape)
    # Посмотрим на слова в словаре
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
    # float_formatter = lambda x: "%.2f" % x
    # np.set_printoptions(formatter={'float_kind': float_formatter})
    return matrix.toarray(), vectorizer.get_feature_names()


def print_info_matrix(matrix, vocab):
    print(matrix.shape)
    # print(np.matrix(matrix))
    print(vocab[0:100])
    print_often_words(matrix, vocab)


def read_directory(directory):
    # f = open(directory + "0_10.txt", "r")
    # print(f.read())
    only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    # only_files = ['0_10.txt']
    # row = []
    # matrix = []
    i = 0
    matrix = ["0", "a"]  # TODO remove it
    # size = only_files.index()
    print("start handling files")
    for file in only_files:
        # print(file)
        f = open(directory + file, "r", encoding="utf-8")
        text = f.read()
        star = re.search(PATTERN, file).group(1)
        row = [star, text]
        # print(row)
        # matrix.append(row)
        # TODO Why is is so slow? np.vstack?
        matrix = np.vstack((matrix, row))
        # matrix = np.concatenate((matrix, row))
        i = i + 1
        if (i + 1) % 100 == 0:
            print("Files %d" % (i + 1))
            # break
        if (i + 1) % 1000 == 0:
            break
    return matrix
    # matrix like this = [id star review]


def take_train_data():
    pos = read_directory(DIRECTORY + "pos\\")
    neg = read_directory(DIRECTORY + "neg\\")
    return np.vstack((pos, neg))
    # first = ["8",
    #          "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."]
    # second = ["3",
    #           "The film starts with a manager (Nicholas Bell) giving welcome investors (Robert Carradine) to Primal Park . A secret project mutating a primal animal using fossilized DNA, like ¨Jurassik Park¨, and some scientists resurrect one of nature's most fearsome predators, the Sabretooth tiger or Smilodon . Scientific ambition turns deadly, however, and when the high voltage fence is opened the creature escape and begins savagely stalking its prey - the human visitors , tourists and scientific.Meanwhile some youngsters enter in the restricted area of the security center and are attacked by a pack of large pre-historical animals which are deadlier and bigger . In addition , a security agent (Stacy Haiduk) and her mate (Brian Wimmer) fight hardly against the carnivorous Smilodons. The Sabretooths, themselves , of course, are the real star stars and they are astounding terrifyingly though not convincing. The giant animals savagely are stalking its prey and the group run afoul and fight against one nature's most fearsome predators. Furthermore a third Sabretooth more dangerous and slow stalks its victims.<br /><br />The movie delivers the goods with lots of blood and gore as beheading, hair-raising chills,full of scares when the Sabretooths appear with mediocre special effects.The story provides exciting and stirring entertainment but it results to be quite boring .The giant animals are majority made by computer generator and seem totally lousy .Middling performances though the players reacting appropriately to becoming food.Actors give vigorously physical performances dodging the beasts ,running,bound and leaps or dangling over walls . And it packs a ridiculous final deadly scene. No for small kids by realistic,gory and violent attack scenes . Other films about Sabretooths or Smilodon are the following : ¨Sabretooth(2002)¨by James R Hickox with Vanessa Angel, David Keith and John Rhys Davies and the much better ¨10.000 BC(2006)¨ by Roland Emmerich with with Steven Strait, Cliff Curtis and Camilla Belle. This motion picture filled with bloody moments is badly directed by George Miller and with no originality because takes too many elements from previous films. Miller is an Australian director usually working for television (Tidal wave, Journey to the center of the earth, and many others) and occasionally for cinema ( The man from Snowy river, Zeus and Roxanne,Robinson Crusoe ). Rating : Below average, bottom of barrel."]
    # third = ["7", "asdasdasd asd "]
    # first = ["8", "3"]
    # first = ["id", "review"]
    # second = ["With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter.", "The film starts with a manager (Nicholas Bell) giving welcome investors (Robert Carradine) to Primal Park . A secret project mutating a primal animal using fossilized DNA, like ¨Jurassik Park¨, and some scientists resurrect one of nature's most fearsome predators, the Sabretooth tiger or Smilodon . Scientific ambition turns deadly, however, and when the high voltage fence is opened the creature escape and begins savagely stalking its prey - the human visitors , tourists and scientific.Meanwhile some youngsters enter in the restricted area of the security center and are attacked by a pack of large pre-historical animals which are deadlier and bigger . In addition , a security agent (Stacy Haiduk) and her mate (Brian Wimmer) fight hardly against the carnivorous Smilodons. The Sabretooths, themselves , of course, are the real star stars and they are astounding terrifyingly though not convincing. The giant animals savagely are stalking its prey and the group run afoul and fight against one nature's most fearsome predators. Furthermore a third Sabretooth more dangerous and slow stalks its victims.<br /><br />The movie delivers the goods with lots of blood and gore as beheading, hair-raising chills,full of scares when the Sabretooths appear with mediocre special effects.The story provides exciting and stirring entertainment but it results to be quite boring .The giant animals are majority made by computer generator and seem totally lousy .Middling performances though the players reacting appropriately to becoming food.Actors give vigorously physical performances dodging the beasts ,running,bound and leaps or dangling over walls . And it packs a ridiculous final deadly scene. No for small kids by realistic,gory and violent attack scenes . Other films about Sabretooths or Smilodon are the following : ¨Sabretooth(2002)¨by James R Hickox with Vanessa Angel, David Keith and John Rhys Davies and the much better ¨10.000 BC(2006)¨ by Roland Emmerich with with Steven Strait, Cliff Curtis and Camilla Belle. This motion picture filled with bloody moments is badly directed by George Miller and with no originality because takes too many elements from previous films. Miller is an Australian director usually working for television (Tidal wave, Journey to the center of the earth, and many others) and occasionally for cinema ( The man from Snowy river, Zeus and Roxanne,Robinson Crusoe ). Rating : Below average, bottom of barrel."]
    # matrix = np.vstack((first, second))
    # matrix = np.vstack((matrix, third))
    # return matrix

    # return pd.read_csv(DATA_FOR_LEARNING_FULL, header=0, delimiter="\t", quoting=3)


def column(matrix, i):
    return [row[i] for row in matrix]


def prepare_data():
    train_data = take_train_data()
    print(len(train_data))
    # print(train_data)
    # testing(train_data)
    norm_text = normalization_text_matrix(train_data)
    # print("Len :")
    # print(len(norm_text[1]))
    # print(norm_text[1])
    # print(len(norm_text[2]))
    # print(norm_text[2])
    # print("Tfidf method")
    # [matrix_tfidf, vocab_tfidf] = tfidf_method(norm_text)
    # print_info_matrix(matrix_tfidf, vocab_tfidf)
    print("\nBag of words")
    [matrix, vocab] = bag_of_words(norm_text)
    # print_info_matrix(matrix, vocab)
    # print(matrix[0])
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
