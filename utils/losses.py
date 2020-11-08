import tensorflow.keras.backend as bk
from tensorflow.keras.losses import binary_crossentropy


def bce():
    def bce_loss(y_true, y_pred):
        res = binary_crossentropy(y_true, y_pred)
        return res

    return bce_loss


def wasserstein_loss():
    def wasserstein(y_true, y_pred):
        return bk.mean(y_true * y_pred)

    return wasserstein
