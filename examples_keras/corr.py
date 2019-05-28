import numpy as np
import pickle
import h5py
import prepare_train_data as prepare

BAG_OF_WORDS_FULL = 'bag_of_words_full'
BAG_OF_WORDS_STARS_FULL = 'bag_of_words_stars_full'
BAG_OF_WORDS_VOCAB_FULL = 'bag_of_words_vocab_full'
BAG_OF_WORDS_FULL_AFTER_FS = "full_after_fs_bag_of_words"
BAG_OF_WORDS_STARS_FULL_AFTER_FS = "full_after_fs_bag_of_words_start"
BAG_OF_WORDS_VOCAB_FULL_AFTER_FS = "full_after_fs_bag_of_words_vocab"
BAG_OF_WORDS_SMALL = 'bag_of_words_small'
BAG_OF_WORDS_STARS_SMALL = 'bag_of_words_stars_small'
BAG_OF_WORDS_VOCAB_SMALL = 'bag_of_words_vocab_small'


def run():
    # read nparray, dimensions (102000, 60)

    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_STARS_FULL_AFTER_FS,
    #                                                  BAG_OF_WORDS_VOCAB_FULL_AFTER_FS)
    # matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_FULL,
    #                                                  BAG_OF_WORDS_STARS_FULL,
    #                                                  BAG_OF_WORDS_VOCAB_FULL)
    matrix, stars, vocab = prepare.take_bag_of_words(BAG_OF_WORDS_SMALL,
                                                     BAG_OF_WORDS_STARS_SMALL,
                                                     BAG_OF_WORDS_VOCAB_SMALL)

    print(matrix.shape)
    matrix = matrix.transpose()
    print(matrix.shape)

    x = matrix
    # z-normalize the data -- first compute means and standard deviations
    xave = np.average(x, axis=1)
    xstd = np.std(x, axis=1)

    # transpose for the sake of broadcasting (doesn't seem to work otherwise!)
    ztrans = x.T - xave
    # ztrans = x - xave
    ztrans /= xstd

    # transpose back
    z = ztrans.T
    # z = ztrans
    # compute correlation matrix - shape = (102000, 102000)
    arr = np.matmul(z, z.T)
    arr /= z.shape[0]

    print(arr.shape)
    print(arr)
    # output to HDF5 file
    with h5py.File('correlation_matrix_small.h5', 'w') as hf:
        hf.create_dataset("correlation", data=arr)
    # correlated_features = set()
    # for i in range(len(arr.columns)):
    #     for j in range(i):
    #         if abs(arr.iloc[i, j]) > 0.8:
    #             colname = arr.columns[i]
    #             correlated_features.add(colname)
    correlated_features = []
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if abs(arr[i][j]) > 0.8:
                print(arr[i][j])
                correlated_features.append(arr[i][j])
    print(len(correlated_features))


# https://stackoverflow.com/questions/52427933/how-to-calculate-a-very-large-correlation-matrix
if __name__ == '__main__':
    run()
#     doesn't work
