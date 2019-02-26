import tensorflow as tf
import numpy as np


def main():
    array0 = np.zeros((32, 32, 32, 1))

    array1 = np.ones((32, 32, 32, 1))

    array = np.array([array0, array0, array0, array0, array1, array1, array0, array1])

    array_ = np.array([array0, array0, array0, array0, array1, array1, array0, array1])

    dataset = tf.data.Dataset.from_tensor_slices((array, array_))

    def filter_fn(x1, x2):
        return tf.not_equal(tf.math.reduce_sum(x1, axis=(0, 1, 2, 3)), 0)

    dataset = dataset.filter(filter_fn)
    dataset = dataset.repeat()
    dataset = dataset.batch(2)

    iterator = dataset.make_one_shot_iterator()

    sess = tf.Session()

    element = sess.run(iterator.get_next())

    print("debug")


if __name__ == '__main__':
    main()
