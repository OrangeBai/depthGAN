from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


def activation_layer(x, activation):
    if activation:
        if activation == 'LeakyReLU':
            x = LeakyReLU(alpha=0.1)(x)
        else:
            x = Activation(activation)(x)
    return x


def dense_layer(input_layer, units, activation, batch_norm=True, **kwargs):
    x = Dense(units, **kwargs)(input_layer)

    if batch_norm:
        x = BatchNormalization()(x)

    x = activation_layer(x, activation)
    return x


def conv_layer(x, filters, kernel_size, batch_norm, strides=(1, 1), padding='same', **kwargs):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, **kwargs)(x)
    x = batch_norm()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def conv_transpose_layer(x, filters, kernel_size, activation, strides=1, padding='same', batch_norm=False,
                         drop_out=False, **kwargs):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides, **kwargs)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    x = activation_layer(x, activation)
    if drop_out:
        x = Dropout(0.4)(x)
    return x


def res_block_down_sampling(input_tensor, filters, stride, norm, identity_block):
    x = short_cut = input_tensor

    x = Conv2D(filters, (1, 1), strides=stride, padding='same')(x)
    x = norm()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = norm()(x)

    if not identity_block:
        short_cut = Conv2D(filters, (1, 1), strides=stride, padding='same')(short_cut)
        short_cut = norm()(short_cut)

    x = Add()([x, short_cut])
    x = LeakyReLU(alpha=0.2)(x)
    return x


def res_block(input_tensor, filters, stride, activation, block_name, identity_block, norm, **kwargs):
    """
    Residual block for ResNet
    :param activation:
    :param stride:
    :param filters:
    :param input_tensor: Input tensor
    :param block_name: block name
    :param identity_block:  if True, the block is set to be identity block
                            else, the block is set to be bottleneck block
    :return:Output tensor
    """
    shortcut = input_tensor
    x = conv_layer(input_tensor,
                   filters=filters,
                   kernel_size=(3, 3),
                   strides=stride,
                   activation=activation,
                   name=block_name + '_Conv1',
                   **kwargs)

    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   activation=None,
                   name=block_name + '_Conv2',
                   **kwargs)

    if not identity_block:
        shortcut = conv_layer(shortcut,
                              filters=filters,
                              kernel_size=(1, 1),
                              strides=stride,
                              activation=None,
                              name=block_name + 'shortcut',
                              **kwargs)

    x = Add()([x, shortcut])

    x = activation_layer(x, activation)

    return x


def bn_transpose_block(input_tensor, filters, stride, activation, block_name, identity_block, **kwargs):
    """
    bottle net block for ResNet
    Bottle net block for ResNet
    :param input_tensor: Input tensor
    :param filters:
    :param stride:
    :param activation:
    :param block_name: block name
    :param identity_block: If True, set as identity block
    :return:
    """
    shortcut = input_tensor
    x = conv_transpose_layer(input_tensor,
                             filters=filters,
                             kernel_size=(1, 1),
                             strides=stride,
                             activation=activation,
                             name=block_name + '_Conv1',
                             **kwargs)
    x = conv_transpose_layer(x,
                             filters=filters,
                             kernel_size=(3, 3),
                             activation=activation,
                             name=block_name + '_Conv2',
                             **kwargs)
    x = conv_transpose_layer(x,
                             filters=filters / 2,
                             kernel_size=(1, 1),
                             activation=None,
                             name=block_name + '_Conv3',
                             **kwargs)

    if not identity_block:
        shortcut = conv_transpose_layer(shortcut,
                                        filters=filters / 2,
                                        kernel_size=(1, 1),
                                        strides=stride,
                                        activation=None,
                                        name=block_name + '_shortcut',
                                        **kwargs)

    x = Add()([x, shortcut])

    x = activation_layer(x, activation)
    return x


def bn_block(input_tensor, filters, stride, activation, block_name, identity_block, **kwargs):
    """
    bottle net block for ResNet
    Bottle net block for ResNet
    :param input_tensor: Input tensor
    :param filters:
    :param stride:
    :param activation:
    :param block_name: block name
    :param identity_block: If True, set as identity block
    :return:
    """
    shortcut = input_tensor
    x = conv_layer(input_tensor,
                   filters=filters,
                   kernel_size=(1, 1),
                   strides=stride,
                   activation=activation,
                   name=block_name + '_Conv1',
                   **kwargs)
    x = conv_layer(x,
                   filters=filters,
                   kernel_size=(3, 3),
                   activation=activation,
                   name=block_name + '_Conv2',
                   **kwargs)
    x = conv_layer(x,
                   filters=filters * 2,
                   kernel_size=(1, 1),
                   activation=None,
                   name=block_name + '_Conv3',
                   **kwargs)

    if not identity_block:
        shortcut = conv_layer(shortcut,
                              filters=filters * 2,
                              kernel_size=(1, 1),
                              strides=stride,
                              activation=None,
                              name=block_name + '_shortcut',
                              **kwargs)

    x = Add()([x, shortcut])

    x = activation_layer(x, activation)

    return x


#
# def fast_up_transpose_conv(x, filters, batch_norm):
#     a = conv_transpose_layer(x, filters, (3, 3), batch_norm=False)
#     b = conv_transpose_layer(x, filters, (2, 3), batch_norm=False)
#     c = conv_transpose_layer(x, filters, (3, 2), batch_norm=False)
#     d = conv_transpose_layer(x, filters, (2, 2), batch_norm=False)
#     left = interleave([a, b], axis=1)  # columns
#     right = interleave([c, d], axis=1)  # columns
#     output = interleave([left, right], axis=2)  # rows
#
#     return output


def fast_up_transpose_projection(input_tensor, filters, activation):
    shortcut = fast_up_transpose_conv(input_tensor, filters, None)
    x = fast_up_transpose_conv(input_tensor, filters, activation)
    x = conv_transpose_layer(x, filters, (3, 3), activation=activation, batch_norm=False)

    x = Add()([x, shortcut])
    if activation == 'LeakyReLU':
        x = LeakyReLU(alpha=0.1)(x)
    else:
        x = Activation(activation)(x)
    return x


def fast_upconv(x, filters, norm):
    a = conv_layer(x, filters, (3, 3), batch_norm=norm)
    b = conv_layer(x, filters, (2, 3), batch_norm=norm)
    c = conv_layer(x, filters, (3, 2), batch_norm=norm)
    d = conv_layer(x, filters, (2, 2), batch_norm=norm)
    left = interleave([a, b], axis=1)  # columns
    right = interleave([c, d], axis=1)  # columns
    output = interleave([left, right], axis=2)  # rows

    return output


def fast_up_projection(input_tensor, filters, norm):
    shortcut = fast_upconv(input_tensor, filters, norm)
    x = fast_upconv(input_tensor, filters, norm)
    x = conv_layer(x, filters, (3, 3), batch_norm=norm)

    x = Add()([x, shortcut])
    x = LeakyReLU(alpha=0.2)(x)
    return x


def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return BatchNormalization
    # elif norm == 'instance_norm':
    #     return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return LayerNormalization


def penalized_tanh(alpha):
    def activation(x):
        pos = K.tanh(K.relu(x))
        neg = -alpha * K.tanh(K.relu(-x))
        return pos + neg

    return activation


def enhanced_sigmoid(alpha, beta):
    alpha = alpha + beta * np.random.rand()
    beta = 4 / alpha

    def activation(x):
        return alpha * K.sigmoid(beta * x) - alpha / 2

    return activation
