import numpy as np
from keras import models
from keras import layers
import prepare_train_data as prepare

FILE_MODEL = 'neuronal_sigmoid_model.h5'
BAG_OF_WORDS_FULL_AFTER_FS = "full_after_fs_bag_of_words"
BAG_OF_WORDS_STARS_FULL_AFTER_FS = "full_after_fs_bag_of_words_start"
BAG_OF_WORDS_VOCAB_FULL_AFTER_FS = "full_after_fs_bag_of_words_vocab"


# 18500/18752 [============================>.] - ETA: 0s - loss: 0.2019 - acc: 0.9316
# 18752/18752 [==============================] - 38s 2ms/step
# - loss: 0.2026 - acc: 0.9313 - val_loss: 0.4635 - val_acc: 0.8427
# Test-Accuracy: 0.8515200054645538
# Changed epoch from 2 to 5
# Test-Accuracy: 0.8415200054645538
# Changed epoch = 1
# 18752/18752 [==============================] - 36s 2ms/step -
# loss: 0.4940 - acc: 0.7839 - val_loss: 0.3651 - val_acc: 0.8630
# Test-Accuracy: 0.8630400037765503
def modify_stars_to_bi_optional(stars):
    for i in range(len(stars)):
        if int(stars[i]) > 5:
            stars[i] = 1
        else:
            stars[i] = 0


def run():
    [matrix, stars, vocab] = prepare.take_bag_of_words(BAG_OF_WORDS_FULL_AFTER_FS,
                                                       BAG_OF_WORDS_STARS_FULL_AFTER_FS,
                                                       BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    prepare.print_info_matrix(matrix, vocab)
    num_words = len(matrix[0])
    size_rows = len(matrix)

    modify_stars_to_bi_optional(stars)
    size_tests_data, test_x, test_y, train_x, train_y = divide_train_and_test_data(matrix, size_rows, stars)
    model = create_model(num_words)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    results = model.fit(
        train_x,
        train_y,
        epochs=2,  # 5, 10, 15, 20 maximum
        batch_size=500,
        validation_data=(test_x, test_y)
    )
    score = model.evaluate(test_x, test_y, verbose=0)
    print("Test-Accuracy:", np.mean(results.history["val_acc"]))
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    model.save(FILE_MODEL)


def create_model(num_words):
    print("num_words: " + str(num_words))
    model = models.Sequential()
    model.add(layers.Dense(50, activation="relu", input_shape=(num_words,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(50, activation="relu"))
    # Output- Layer
    model.add(layers.Dense(1, activation="sigmoid"))
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


def use_model():
    new_model = models.load_model(FILE_MODEL)
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    # data = vectorization(data, num_words)
    # test_x, test_y, train_x, train_y = divide_train_and_test_data(data, targets)
    num_words = len(matrix[0])
    size_rows = len(matrix)
    size_tests_data, test_x, test_y, train_x, train_y = divide_train_and_test_data(matrix, size_rows, stars)
    print(train_y[1:2])
    res = new_model.predict_classes(train_x[1:2, :])
    print(res)
