import numpy as np
from tensorflow.keras.datasets import cifar10
import tensorflow as tf


def cifar_10_gen(batch_size=32):
    (x1, y1), (x2, y2) = cifar10.load_data()

    x = np.concatenate((x1, x2), axis=0)
    x = x / 127.5 - 1
    y = np.concatenate((y1, y2), axis=0)

    x = [x[idx:idx + batch_size] for idx in range(0, len(x), batch_size)]
    y = [y[idx:idx + batch_size] for idx in range(0, len(y), batch_size)]

    data_length = len(x)
    while True:
        id = np.random.choice(data_length, 1)[0]
        yield tf.convert_to_tensor(x[id], dtype=float), tf.convert_to_tensor(y[id])
