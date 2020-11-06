import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


def discriminator_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

    return model


model = discriminator_model()

model.summary()

x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5


def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_real_samples(dataset, n_samples):
    # define random instances
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate class label (label = 1)
    y = np.ones((n_samples, 1))
    return x, y


# generate n fake samples with class label

def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # generate class label (label = 0)
    y = np.zeros((n_samples, 1))
    return x, y


def generator_model(latent_dim):
    n_nodes = 256 * 4 * 4
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=n_nodes, input_dim=latent_dim),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Reshape((4, 4, 256)),
        # upsample to 8*8
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # upsample to 16*16
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # upsample to 32*32
        tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), padding='same', strides=(2, 2)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # output layer
        tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), activation='tanh', padding='same')
    ])
    return model


model = generator_model(100)

model.summary()


def gan_model(g_model, d_model):
    # freeze discriminator model
    d_model.trainable = False

    model = tf.keras.models.Sequential([
        g_model,
        d_model
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

    return model


latent_dim = 100

g_model = generator_model(latent_dim)

d_model = discriminator_model()

gan_model = gan_model(g_model, d_model)

gan_model.summary()


def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # make plot
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.imshow(examples[i])

    # save plots
    filename = r'C:\Users\jzp0306\Desktop\generated_plot_e%03d.png' % (epoch + 1)
    plt.savefig(filename)


# evaluate discriminator model performance, display generated images, save generator model

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    x_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real samples
    _, acc_real = d_model.evaluate(x_real, y_real, verbose=0)
    # prepare fake samples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake samples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # display discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))

    # show and save the plots of generated images
    save_plot(x_fake, epoch)

    # save generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


def train_gan(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=128):
    bat_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches
        for j in range(bat_per_epoch):
            # randomly select n real samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            # update standalone discriminator model
            d_loss1, _ = d_model.train_on_batch(x_real, y_real)
            # generate fake samples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update standalone discriminator model again
            d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)
            # generate points in latent space as the inputs of generator model
            x_gan = generate_latent_points(latent_dim, n_batch)
            # generate class label for fake samples (label = 1)
            y_gan = np.ones((n_batch, 1))
            # update the generator model with discriminator model errors
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            # display the loss
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epoch, d_loss1, d_loss2, g_loss))

        # evaluate model performance every 5 epochs
        if (i + 1) % 5 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# train gan model

train_gan(g_model, d_model, gan_model, x_train, latent_dim)

model = tf.keras.models.load_model('generator_model_020.h5')  # load model saved after 20 epochs

latent_points = generate_latent_points(100, 100)  # generate points in latent space

X = model.predict(latent_points)  # generate images

X = (X + 1) / 2.0  # scale the range from [-1,1] to [0,1]

fig = plt.gcf()
fig.set_size_inches(20, 20)
for i in range(100):
    plt.subplot(10, 10, 1 + i)
    plt.imshow(X[i])
