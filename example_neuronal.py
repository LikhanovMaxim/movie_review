# https://neurohive.io/ru/tutorial/nejronnaya-set-keras-python/
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import prepare_train_data as prepare
from keras.models import Sequential
from keras.layers import Dense, Activation


# Пришло время подготовить данные. Нужно векторизовать каждый обзор и заполнить его нулями,
# чтобы вектор содержал ровно 10 000 чисел. Это означает, что каждый обзор,
# который короче 10 000 слов, мы заполняем нулями. Это делается потому,
# что самый большой обзор имеет почти такой же размер, а каждый элемент
# входных данных нашей нейронной сети должен иметь одинаковый размер.
def vectorize(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def column(matrix, i):
    return [row[i] for row in matrix]


def prepare_ska():
    # first = ["8",
    #          "With all this stuff going down at the moment with MJ i've started listening to his music, watching the odd documentary here and there, watched The Wiz and watched Moonwalker again. Maybe i just want to get a certain insight into this guy who i thought was really cool in the eighties just to maybe make up my mind whether he is guilty or innocent. Moonwalker is part biography, part feature film which i remember going to see at the cinema when it was originally released. Some of it has subtle messages about MJ's feeling towards the press and also the obvious message of drugs are bad m'kay.<br /><br />Visually impressive but of course this is all about Michael Jackson so unless you remotely like MJ in anyway then you are going to hate this and find it boring. Some may call MJ an egotist for consenting to the making of this movie BUT MJ and most of his fans would say that he made it for the fans which if true is really nice of him.<br /><br />The actual feature film bit when it finally starts is only on for 20 minutes or so excluding the Smooth Criminal sequence and Joe Pesci is convincing as a psychopathic all powerful drug lord. Why he wants MJ dead so bad is beyond me. Because MJ overheard his plans? Nah, Joe Pesci's character ranted that he wanted people to know it is he who is supplying drugs etc so i dunno, maybe he just hates MJ's music.<br /><br />Lots of cool things in this like MJ turning into a car and a robot and the whole Speed Demon sequence. Also, the director must have had the patience of a saint when it came to filming the kiddy Bad sequence as usually directors hate working with one kid let alone a whole bunch of them performing a complex dance scene.<br /><br />Bottom line, this movie is for people who like MJ on one level or another (which i think is most people). If not, then stay away. It does try and give off a wholesome message and ironically MJ's bestest buddy in this movie is a girl! Michael Jackson is truly one of the most talented people ever to grace this planet but is he guilty? Well, with all the attention i've gave this subject....hmmm well i don't know because people can be different behind closed doors, i know this for a fact. He is either an extremely nice but stupid guy or one of the most sickest liars. I hope he is not the latter."]
    # second = ["3",
    #           "The film starts with a manager (Nicholas Bell) giving welcome investors (Robert Carradine) to Primal Park . A secret project mutating a primal animal using fossilized DNA, like ¨Jurassik Park¨, and some scientists resurrect one of nature's most fearsome predators, the Sabretooth tiger or Smilodon . Scientific ambition turns deadly, however, and when the high voltage fence is opened the creature escape and begins savagely stalking its prey - the human visitors , tourists and scientific.Meanwhile some youngsters enter in the restricted area of the security center and are attacked by a pack of large pre-historical animals which are deadlier and bigger . In addition , a security agent (Stacy Haiduk) and her mate (Brian Wimmer) fight hardly against the carnivorous Smilodons. The Sabretooths, themselves , of course, are the real star stars and they are astounding terrifyingly though not convincing. The giant animals savagely are stalking its prey and the group run afoul and fight against one nature's most fearsome predators. Furthermore a third Sabretooth more dangerous and slow stalks its victims.<br /><br />The movie delivers the goods with lots of blood and gore as beheading, hair-raising chills,full of scares when the Sabretooths appear with mediocre special effects.The story provides exciting and stirring entertainment but it results to be quite boring .The giant animals are majority made by computer generator and seem totally lousy .Middling performances though the players reacting appropriately to becoming food.Actors give vigorously physical performances dodging the beasts ,running,bound and leaps or dangling over walls . And it packs a ridiculous final deadly scene. No for small kids by realistic,gory and violent attack scenes . Other films about Sabretooths or Smilodon are the following : ¨Sabretooth(2002)¨by James R Hickox with Vanessa Angel, David Keith and John Rhys Davies and the much better ¨10.000 BC(2006)¨ by Roland Emmerich with with Steven Strait, Cliff Curtis and Camilla Belle. This motion picture filled with bloody moments is badly directed by George Miller and with no originality because takes too many elements from previous films. Miller is an Australian director usually working for television (Tidal wave, Journey to the center of the earth, and many others) and occasionally for cinema ( The man from Snowy river, Zeus and Roxanne,Robinson Crusoe ). Rating : Below average, bottom of barrel."]
    # res = np.vstack((first, second))
    # print(res)
    prepare.prepare_data()


def run():
    num_words = 10000
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    data = np.concatenate((training_data, testing_data), axis=0)

    targets = np.concatenate((training_targets, testing_targets), axis=0)
    data = vectorize(data, num_words)
    # Также нужно выполнить преобразование переменных в тип float.
    targets = np.array(targets).astype("float32")
    # Разделим датасет на обучающий и тестировочный наборы.
    # Обучающий набор будет состоять из 40 000 обзоров,
    # тестировочный — из 10 000.
    size_tests_data = 10000
    train_x = data[size_tests_data:]
    train_y = targets[size_tests_data:]
    test_x = data[:size_tests_data]
    test_y = targets[:size_tests_data]
    # Последовательная модель
    model = models.Sequential()
    # Input - Layer
    # На каждом слое используется функция «dense» для полного соединения слоев друг с другом.
    model.add(layers.Dense(50, activation="relu", input_shape=(num_words,)))
    # Hidden - Layers. Dropout
    # Обратите внимание, что вы всегда должны использовать коэффициент исключения в диапазоне от 20% до 50%.
    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()
    # compiling the model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    results = model.fit(
        train_x, train_y,
        epochs=2,
        batch_size=500,
        validation_data=(test_x, test_y)
    )
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))


def example_second():
    # For a single-input model with 10 classes (categorical classification):

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=100))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Generate dummy data
    import numpy as np
    data = np.random.random((1000, 100))
    labels = np.random.randint(10, size=(1000, 1))
    # <class 'tuple'>: (1000, 1) ndarray
    # Convert labels to categorical one-hot encoding
    one_hot_labels = to_categorical(labels, num_classes=10)

    # Train the model, iterating on the data in batches of 32 samples
    model.fit(data, one_hot_labels, epochs=10, batch_size=32)


if __name__ == '__main__':
    run()
    # prepare_ska()
    # run_my()
    # example_second()
