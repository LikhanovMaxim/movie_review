import prepare_train_data as prepare
import neuronal as neuronal


def prepare_data():
    prepare.prepare_data_by_bag_of_words()
    print("Start reading")
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    print(stars)
    print(stars.shape)


if __name__ == '__main__':
    # prepare.prepare_data_by_bag_of_words()
    neuronal.run()
