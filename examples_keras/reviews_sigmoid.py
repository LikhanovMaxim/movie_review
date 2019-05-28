# https://neurohive.io/ru/tutorial/nejronnaya-set-keras-python/
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
# import prepare_train_data as prepare
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import optimizers

# from tensorflow import keras
# from tensorflow.keras import layers

# Пришло время подготовить данные. Нужно векторизовать каждый обзор и заполнить его нулями,
# чтобы вектор содержал ровно 10 000 чисел. Это означает, что каждый обзор,
# который короче 10 000 слов, мы заполняем нулями. Это делается потому,
# что самый большой обзор имеет почти такой же размер, а каждый элемент
# входных данных нашей нейронной сети должен иметь одинаковый размер.
FILE_MODEL = 'reviews_sigmoid_model.h5'


def vectorization(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# def column(matrix, i):
#     return [row[i] for row in matrix]


def take_data():
    num_words = 10000
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=num_words)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)
    # Также нужно выполнить преобразование переменных в тип float.
    targets = np.array(targets).astype("float32")
    return data, num_words, targets


def run():
    data, num_words, targets = take_data()
    data = vectorization(data, num_words)
    # Разделим датасет на обучающий и тестировочный наборы.
    # Обучающий набор будет состоять из 40 000 обзоров,
    # тестировочный — из 10 000.
    test_x, test_y, train_x, train_y = divide_train_and_test_data(data, targets)
    model = create_model(num_words)
    # compiling the model
    # Будем использовать оптимизатор «adam».
    # Оптимизатор — это алгоритм, который изменяет веса и смещения во время обучения.
    # В качестве функции потерь используем бинарную кросс-энтропию (так как мы работаем с бинарной классификацией),
    # в качестве метрики оценки — точность.

    # TODO sgd vs adam
    # TODO decay & momentum
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    # learn neuronal network
    # Мы будем делать это с размером партии 500 и только двумя эпохами,
    # поскольку я выяснил, что модель начинает переобучаться, если тренировать ее дольше.
    # Размер партии определяет количество элементов, которые будут распространяться по сети,
    # а эпоха — это один проход всех элементов датасета.
    # Обычно больший размер партии приводит к более быстрому обучению, но не всегда — к быстрой сходимости.
    # Меньший размер партии обучает медленнее, но может быстрее сходиться.
    # Выбор того или иного варианта определенно зависит от типа решаемой задачи, и лучше попробовать каждый из них.
    # Если вы новичок в этом вопросе, я бы посоветовал вам сначала использовать размер партии 32,
    # что является своего рода стандартом.
    results = model.fit(
        train_x, train_y,
        epochs=2,
        batch_size=32,
        validation_data=(test_x, test_y)
    )
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))
    history = results
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Save the model
    model.save(FILE_MODEL)
    # keras.experimental.export_saved_model(model, FILE_MODEL)


def create_model(num_words):
    # Последовательная модель https://keras.io/models/sequential/
    model = models.Sequential()
    # Input - Layer
    # На каждом слое используется функция «dense» для полного соединения слоев друг с другом.
    # Обратите внимание, что мы устанавливаем размер входных элементов датасета равным 10 000,
    # потому что наши обзоры имеют размер до 10 000 целых чисел.
    # Входной слой принимает элементы с размером 10 000, а выдает — с размером 50.
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
    return model


def divide_train_and_test_data(data, targets):
    size_tests_data = 10000
    train_x = data[size_tests_data:]
    train_y = targets[size_tests_data:]
    test_x = data[:size_tests_data]
    test_y = targets[:size_tests_data]
    return test_x, test_y, train_x, train_y


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


def learn_data():
    data, num_words, targets = take_data()
    print("Number of unique words:", len(np.unique(np.hstack(data))))
    print("Categories:", np.unique(targets))
    length = [len(i) for i in data]
    print("Average Review length:", np.mean(length))
    print("Standard Deviation:", round(np.std(length)))
    print("Example:")
    print("Label:", targets[0])
    print(data[0])
    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])
    print(decoded)


def use_model():
    new_model = models.load_model(FILE_MODEL)
    data, num_words, targets = take_data()
    data = vectorization(data, num_words)
    test_x, test_y, train_x, train_y = divide_train_and_test_data(data, targets)
    print(train_x.shape)
    # test = [][train_x[0]]
    # b = test.reshape(10000, 1)
    # a = np.reshape(test, )
    # print(test)
    # print(test.shape)
    # train_x[:1, :].shape
    res = new_model.predict_classes(test_x[:1, :])
    # res = new_model.predict_classes(test)
    print(res)
    # new_model.
    # new_model = keras.experimental.load_from_saved_model(FILE_MODEL)


if __name__ == '__main__':
    run()
    # use_model()
    # learn_data()
    # example_second()
