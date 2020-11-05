import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import *
from config import *
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.losses import binary_crossentropy
from utils.losses import *
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)

img_size = 32
noise_size = 4096
batch_size = 50
classes = 10

(x1, y1), (x2, y2) = cifar10.load_data()

x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)


def generator():
    noise = Input(shape=(noise_size,))
    label = Input(shape=(1,))

    label_embedding = Flatten()(Embedding(classes, noise_size)(label))

    model_input = multiply([noise, label_embedding])

    x = Reshape((4, 4, 256))(model_input)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(0.1)(x)

    x = Conv2DTranspose(256, (5, 5), padding='same', strides=2)(x)
    x = Conv2DTranspose(256, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2DTranspose(128, (5, 5), padding='same', strides=2)(x)
    x = Conv2DTranspose(128, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2DTranspose(64, (5, 5), padding='same', strides=2)(x)
    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    # x = Conv2DTranspose(64, (5, 5), padding='same', strides=2)(x)
    # x = BatchNormalization(momentum=0.9)(x)
    # x = LeakyReLU(0.1)(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', strides=1)(x)
    x = Conv2DTranspose(3, (5, 5), padding='same')(x)
    img = Activation('tanh')(x)

    return Model([noise, label], img)


def discriminator():
    img = Input(shape=(img_size, img_size, 3))

    # x = GaussianNoise(0.1)(img)

    x = Conv2D(64, (3, 3), padding='same', strides=2)(img)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (3, 3), padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # x = Conv2D(512, (3, 3), padding='same', strides=2)(x)
    # x = BatchNormalization(momentum=0.9)(x)
    # x = LeakyReLU(0.2)(x)

    label = Input(shape=(1,))
    label_embedding = Flatten()(Embedding(classes, noise_size)(label))

    flat_img = Flatten()(x)
    model_input = multiply([flat_img, label_embedding])

    nn = Dropout(0.3)(model_input)

    validity = Dense(1, activation='sigmoid')(nn)

    return Model([img, label], validity)


d_model = discriminator()
d_model.compile(loss=bce(), optimizer=Adam(lr=0.002, beta_1=0.5))
d_model.trainable = False
g_model = generator()

noise = Input(shape=(noise_size,))
label = Input(shape=(1,))
img = g_model([noise, label])

valid = d_model([img, label])

combined = Model([noise, label], valid)
combined.compile(loss=bce(), optimizer=Adam(lr=0.0001, beta_1=0.5))


def train(epochs):
    for epoch in range(epochs):

        random = np.random.randint(0, 11)

        for index in range(int(x.shape[0] / batch_size)):

            valid = np.ones((batch_size, 1)) - (np.random.random() * 0.1)
            fake = np.zeros((batch_size, 1)) + (np.random.random() * 0.1)

            x_train = x[index * batch_size: (index + 1) * batch_size]
            y_train = y[index * batch_size: (index + 1) * batch_size]
            x_train = (x_train - 127.5) / 127.5

            if index % 100 == random:
                valid = np.zeros((batch_size, 1)) + (np.random.random() * 0.1)
                fake = np.ones((batch_size, 1)) - (np.random.random() * 0.1)


            noise = np.random.randn(batch_size, noise_size)
            gen_img = g_model.predict([noise, y_train])

            d_model.trainable = True
            d_loss_real = d_model.train_on_batch([x_train, y_train], valid)
            d_loss_fake = d_model.train_on_batch([gen_img, y_train], fake)
            d_loss = 0.5 * (np.add(d_loss_real, d_loss_fake))

            sample_label = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            valid = np.ones((batch_size, 1))
            d_model.trainable = False
            g_loss = combined.train_on_batch([noise, sample_label], valid)

            if index % 500 == 0:
                # print(index)
                # print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
                # gen_img = g_model.predict([noise, y_train])
                # for idx, image in enumerate(gen_img[:10]):
                #     image_path = os.path.join(test_dir, 'Epoch{0}_{1}_{2}.jpg'.format(epoch, index, idx))
                #     image = image * 127.5 + 127.5
                #     cv2.imwrite(image_path, image)

                print(index)
                print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
                sample_images(epoch)

        name = './weights/combined_' + str(epoch) + '.h5'
        combined.save_weights(name)

        # time.sleep(30)


def sample_images(epoch):
    r = 2
    c = 5
    noise = np.random.randn(10, noise_size)
    sample_label = np.arange(0, 10).reshape(-1, 1)

    gen_img = g_model.predict([noise, sample_label])

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            img = image.array_to_img(gen_img[cnt])
            axs[i, j].imshow(img)
            axs[i, j].set_title("Class: %d" % sample_label[cnt])
            axs[i, j].axis('off')
            cnt += 1
    path = os.path.join(test_dir, "%d.png" % epoch)
    fig.savefig(path)
    plt.close()


train(10)
