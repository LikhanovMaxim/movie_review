# TODO find and analyze
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense


def run():
    nb_classes = 10

    # the data, shuffled and split between tran and test sets
    X_test, X_train, y_test, y_train = take_data()

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()
    model.add(Dense(output_dim=nb_classes, input_dim=784, activation='softmax'))

    model.summary()  # Print model info
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=128,
              nb_epoch=5,
              verbose=2,  # how to print
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


def take_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    X_train /= 255
    X_test /= 255
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    return X_test, X_train, y_test, y_train


if __name__ == '__main__':
    run()