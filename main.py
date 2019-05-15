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


def prepare_data():
    # small batch, for test:
    prepare.prepare_data_by_bag_of_words()
    # use it when you want to prepare full bag of words
    # prepare.prepare_data_by_bag_of_words(False, True)


if __name__ == '__main__':
    # neuronal.run()
    neuronal_sigmoid.run()
    # take_data()
    # prepare_data()
