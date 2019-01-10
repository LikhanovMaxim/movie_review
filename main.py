import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import prepare_train_data as prepare
from keras.models import Sequential
from keras.layers import Dense, Activation


def prepare_data():
    prepare.prepare_data()
    print("Start reading")
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    print(stars)
    print(stars.shape)


def change(train_y, size):
    train_y = [int(i) for i in train_y]
    train_y = np.asarray(train_y)
    train_y = np.reshape(train_y, (size, 1))
    train_y[0] = 5
    train_y[1] = 6
    train_y[2] = 0
    return train_y


def run():
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    num_words = len(matrix[0])
    size_rows = len(matrix)
    size_tests_data = int(size_rows / 4)
    size_tests_data_del_2 = int(size_tests_data / 2)
    size_before_smth = (size_rows - size_tests_data_del_2)
    train_x = matrix[size_tests_data_del_2:size_before_smth]
    train_y = stars[size_tests_data_del_2:size_before_smth]
    test_x = np.concatenate((matrix[0:size_tests_data_del_2], matrix[size_before_smth:size_rows]))
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
    one_hot_labels = to_categorical(train_y, num_classes=11)

    test_y = change(test_y, size_tests_data)
    one_hot_labels_2 = to_categorical(test_y, num_classes=11)

    results = model.fit(
        train_x, one_hot_labels,
        epochs=5,  # 10, 15, 20 maximum
        batch_size=32,
        validation_data=(test_x, one_hot_labels_2)
    )
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))


if __name__ == '__main__':
    run()
