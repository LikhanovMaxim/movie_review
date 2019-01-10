import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import prepare_train_data as prepare
from keras.models import Sequential
from keras.layers import Dense, Activation


def prepare_my():
    # Do it once:
    prepare.prepare_data()
    print("Start reading")
    # f = open(DIRECTORY + BAG_OF_WORDS, "r")
    # contents = f.read()
    # print(contents)
    # print(contents.toarray)
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    print(stars)
    print(stars.shape)
    # prepare.print_info_matrix(matrix, vocab)


def change(train_y, size):
    # TODO
    train_y = [int(i) for i in train_y]
    train_y = np.asarray(train_y)
    train_y = np.reshape(train_y, (size, 1))
    train_y[0] = 5
    train_y[1] = 6
    train_y[2] = 0
    return train_y


def run_my():
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    # stars = column(matrix, 0)
    print("stars")
    # print(stars)
    num_words = len(matrix[0])
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    # data = np.concatenate((training_data, testing_data), axis=0)

    # targets = np.concatenate((training_targets, testing_targets), axis=0)
    # data = vectorize(data, num_words)
    # Также нужно выполнить преобразование переменных в тип float.
    # stars = np.array(stars).astype("float32")
    # Разделим датасет на обучающий и тестировочный наборы.
    # Обучающий набор будет состоять из 40 000 обзоров,
    # тестировочный — из 10 000.
    size_rows = len(matrix)
    size_tests_data = int(size_rows / 4)
    size_tests_data_del_2 = int(size_tests_data / 2)
    print("size")
    print(size_rows)
    print(size_tests_data)
    # train_x = matrix[size_tests_data:]
    size_before_smth = (size_rows - size_tests_data_del_2)
    train_x = matrix[size_tests_data_del_2:size_before_smth]
    # train_y = stars[size_tests_data:]
    train_y = stars[size_tests_data_del_2:size_before_smth]
    # test_x = matrix[:size_tests_data]
    test_x = np.concatenate((matrix[0:size_tests_data_del_2], matrix[size_before_smth:size_rows]))
    # test_y = stars[:size_tests_data]
    test_y = np.concatenate((stars[0:size_tests_data_del_2], stars[size_before_smth:size_rows]))
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
    model.add(layers.Dense(11, activation="softmax"))
    model.summary()
    # compiling the model
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",  # binary_crossentropy
        metrics=["accuracy"]
    )
    # Convert labels to categorical one-hot encoding

    train_y = change(train_y, size_rows - size_tests_data)
    # np.reshape(test_y, (50, 1))
    # labels = np.random.randint(10, size=(150, 1))
    # one_hot_labels_2 = to_categorical(test_y, num_classes=8)

    one_hot_labels = to_categorical(train_y, num_classes=11)
    # <class 'tuple'>: (1000, 1) ndarray
    # Convert labels to categorical one-hot encoding
    # one_hot_labels = to_categorical(labels, num_classes=10)
    # <class 'tuple'>: (1000, 1) ndarray
    # Convert labels to categorical one-hot encoding
    test_y = change(test_y, size_tests_data)
    # labels_2 = np.random.randint(10, size=(50, 1))
    # one_hot_labels_2 = to_categorical(labels_2, num_classes=10)
    one_hot_labels_2 = to_categorical(test_y, num_classes=11)
    results = model.fit(
        train_x, one_hot_labels,
        epochs=5,  # 10, 15, 20 maximum
        batch_size=32,
        validation_data=(test_x, one_hot_labels_2)
    )
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))


if __name__ == '__main__':
    # prepare_my()
    run_my()
