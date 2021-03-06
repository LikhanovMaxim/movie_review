import prepare_train_data as prepare
import neuronal as neuronal
import neuronal_sigmoid as neuronal_sigmoid


# it is for test
def take_data():
    print("Start reading")
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    prepare.print_info_matrix(matrix, vocab)
    print(stars)
    print(stars.shape)


#   so we get matrix for 25000 * 66444
#                   25000 - count of reviews: 12500 pos + 12500 neg
#                   66444 - count of unique words TODO чет много
#  More often words:
#       51694 movie
#       47034 film
#       27741 one
#       20740 like
#       15959 time
#       15197 good
#       14177 character
#       13165 story
#       12646 even
#       12514 get

def prepare_data_tf_idf():
    # small_batch_for_test()
    prepare.prepare_data_by_tf_idf(False, True)


def small_batch_for_test():
    prepare.prepare_data_by_tf_idf()
    matrix, stars, vocab = prepare.take_tf_idf()
    prepare.print_info_matrix(matrix, vocab)
    # a = np.nonzero(matrix[1] > 0)
    for i in range(len(matrix[1])):
        if matrix[1][i] > 0:
            print(i)
            print(vocab[i])
            print(matrix[1][i])
    # sorted(matrix[1], reverse=True)
    # print(a)


def prepare_data_bag_of_word():
    # small batch, for test:
    prepare.prepare_data_by_bag_of_words(True, True)
    # use it when you want to prepare full bag of words
    # prepare.prepare_data_by_bag_of_words(False, True)


if __name__ == '__main__':
    [matrix, stars, vocab] = prepare.take_bag_of_words()
    # [matrix, stars, vocab] = prepare.take_tf_idf()
    neuronal.run(matrix, stars, vocab)
    # neuronal_sigmoid.run()
    # take_data()
    # prepare_data_bag_of_word()
    # prepare_data_tf_idf()
