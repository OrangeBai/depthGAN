from models.nets import *


def res20(input_tensor):
    x = conv_layer(input_tensor, 16, (3, 3), name='Conv1')

    x = res_block(x, 32, (2, 2), 'block1_1', identity_block=False)
    x = res_block(x, 32, (1, 1), 'block1_2', identity_block=True)
    x = res_block(x, 32, (1, 1), 'block1_3', identity_block=True)

    x = res_block(x, 64, (2, 2), 'block2_1', identity_block=False)
    x = res_block(x, 64, (1, 1), 'block2_2', identity_block=True)
    x = res_block(x, 64, (1, 1), 'block2_3', identity_block=True)

    x = res_block(x, 128, (2, 2), 'block3_1', identity_block=False)
    x = res_block(x, 128, (1, 1), 'block3_2', identity_block=True)
    x = res_block(x, 128, (1, 1), 'block3_3', identity_block=True)

    return x
