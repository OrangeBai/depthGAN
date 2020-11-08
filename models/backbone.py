from models.nets import *


def res14(input_tensor, activation='LeakyReLU'):
    x = conv_layer(input_tensor, 32, (3, 3), activation, name='Conv1', batch_norm=False)

    x = res_block(x, 32, (2, 2), activation, 'block1_1', identity_block=False)
    x = res_block(x, 32, (1, 1), activation, 'block1_2', identity_block=True)

    x = res_block(x, 64, (2, 2), activation, 'block2_1', identity_block=False)
    x = res_block(x, 64, (1, 1), activation, 'block2_2', identity_block=True)

    x = res_block(x, 128, (2, 2), activation, 'block3_1', identity_block=False)
    patch = res_block(x, 128, (1, 1), activation, 'block3_2', identity_block=True)

    x = res_block(x, 256, (2, 2), activation, 'block4_1', identity_block=False)
    x = res_block(x, 256, (1, 1), activation, 'block4_2', identity_block=True)

    # x = res_block(x, 256, (2, 2), activation, 'block5_1', identity_block=False)
    # x = res_block(x, 256, (2, 2), activation, 'block5_2', identity_block=True)

    return x, patch


def res20(input_tensor):
    x = conv_layer(input_tensor, 16, (3, 3), 'LeakyReLU', name='Conv1')

    x = res_block(x, 32, (2, 2), 'LeakyReLU', 'block1_1', identity_block=False)
    x = res_block(x, 32, (1, 1), 'LeakyReLU', 'block1_2', identity_block=True)
    x = res_block(x, 32, (1, 1), 'LeakyReLU', 'block1_3', identity_block=True)

    x = res_block(x, 64, (2, 2), 'LeakyReLU', 'block2_1', identity_block=False)
    x = res_block(x, 64, (1, 1), 'LeakyReLU', 'block2_2', identity_block=True)
    x = res_block(x, 64, (1, 1), 'LeakyReLU', 'block2_3', identity_block=True)

    x = res_block(x, 128, (2, 2), 'LeakyReLU', 'block3_1', identity_block=False)
    x = res_block(x, 128, (1, 1), 'LeakyReLU', 'block3_2', identity_block=True)
    x = res_block(x, 128, (1, 1), 'LeakyReLU', 'block3_3', identity_block=True)

    return x


def res50(input_tensor, activation, **kwargs):
    x = conv_layer(input_tensor, 64, (7, 7), strides=(2, 2), activation=activation)
    x = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(x)

    x = bn_block(x, 64, (1, 1), activation, 'block_1_1', False, **kwargs)
    x = bn_block(x, 64, (1, 1), activation, 'block_1_2', False, **kwargs)
    x = bn_block(x, 64, (1, 1), activation, 'block_1_3', False, **kwargs)

    x = bn_block(x, 128, (2, 2), activation, 'block_2_1', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_2', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_3', False, **kwargs)
    x = bn_block(x, 128, (1, 1), activation, 'block_2_4', False, **kwargs)

    x = bn_block(x, 256, (2, 2), activation, 'block_3_1', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_2', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_3', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_4', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_5', False, **kwargs)
    x = bn_block(x, 256, (1, 1), activation, 'block_3_6', False, **kwargs)

    x = bn_block(x, 512, (2, 2), activation, 'block_4_1', False, **kwargs)
    x = bn_block(x, 512, (1, 1), activation, 'block_4_2', False, **kwargs)
    x = bn_block(x, 512, (1, 1), activation, 'block_4_3', False, **kwargs)

    return x
