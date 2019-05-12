import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import prepare_train_data as prepare
from keras.models import Sequential
from keras.layers import Dense, Activation


def change(train_y, size):
    train_y = [int(i) for i in train_y]
    train_y = np.asarray(train_y)
    train_y = np.reshape(train_y, (size, 1))
    train_y[0] = 5
    train_y[1] = 6
    train_y[2] = 0
    return train_y


# First layer input = 66444
# 18720/18752 [============================>.] - ETA: 0s - loss: 0.8773 - acc: 0.6792
# 18752/18752 [==============================] - 38s 2ms/step - loss: 0.8772 - acc: 0.6791 -
#                                                           val_loss: 2.0436 - val_acc: 0.3738
# Test-Accuracy: 0.40496000002098087
# Changed batch_size from 32 to 500
# Test-Accuracy: 0.4069759998321533
# Changed hidden_layer from 50 to 500
# Test-Accuracy: 0.3542720012664795
def run():
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    num_words = len(matrix[0])
    size_rows = len(matrix)
    size_tests_data, test_x, test_y, train_x, train_y = divide_train_and_test_data(matrix, size_rows, stars)
    model = create_model(num_words)
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
        train_x,
        one_hot_labels,
        epochs=5,  # 5, 10, 15, 20 maximum
        batch_size=500,
        validation_data=(test_x, one_hot_labels_2)
    )
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))


def create_model(num_words):
    print("num_words: " + str(num_words))
    # Последовательная модель
    model = models.Sequential()
    # Input - Layer
    # На каждом слое используется функция «dense» для полного соединения слоев друг с другом.
    hidden_layer = 50
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
    return model


def divide_train_and_test_data(matrix, size_rows, stars):
    size_tests_data = int(size_rows / 4)
    size_tests_data_del_2 = int(size_tests_data / 2)
    size_before_smth = (size_rows - size_tests_data_del_2)
    train_x = matrix[size_tests_data_del_2:size_before_smth]
    train_y = stars[size_tests_data_del_2:size_before_smth]
    test_x = np.concatenate((matrix[0:size_tests_data_del_2], matrix[size_before_smth:size_rows]))
    test_y = np.concatenate((stars[0:size_tests_data_del_2], stars[size_before_smth:size_rows]))
    return size_tests_data, test_x, test_y, train_x, train_y
