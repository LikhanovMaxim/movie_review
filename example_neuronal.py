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
