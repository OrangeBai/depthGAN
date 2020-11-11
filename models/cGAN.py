from models.nets import *
from models.base_model import *
from models.backbone import *
from numpy.random import random, randint
from utils.losses import *
import cv2
import functools
import tqdm
import imlib as im
import tf2lib as tl


class ConditionalGAN:
    def __init__(self, noise_unit, input_size, image_size, dim, class_number, cgan, penalty_mode='wgan-gp', penalty_weight=10, batch_size=32):
        super().__init__()
        self.noise_units = noise_unit
        self.input_size = input_size
        self.image_size = image_size
        self.dim = dim

        self.class_number = class_number
        self.batch_size = batch_size

        self.penalty_mode = penalty_mode
        self.penalty_weight = penalty_weight

        self.generator = None
        self.discriminator = None

        self.g_optimizer = None
        self.d_optimizer = None

        self.d_loss_fn = None
        self.g_loss_fn = None

        self.cgan = cgan

    def build_generator(self, name='ConvGenerator'):

        n_upsamplings = int(np.log2(self.image_size) - np.log2(self.input_size))

        norm = self.get_norm_mode()
        Norm = get_norm_layer(norm)

        # 0
        x = inputs = Input(shape=(self.noise_units,))
        if self.cgan:
            label_input = Input((1,))
            label_embedding = Flatten()(Embedding(self.class_number, self.noise_units, trainable=True)(label_input))
            x = multiply([x, label_embedding])
            inputs = [inputs, label_input]

        x = Reshape((1, 1, self.noise_units))(x)

        # 1: 1x1 -> 4x4
        d = min(self.dim * 2 ** (n_upsamplings - 1), self.dim * 8)
        x = Conv2DTranspose(d, self.input_size, strides=1, padding='valid', use_bias=False)(x)
        x = Norm()(x)
        x = tf.nn.relu(x)  # or h = keras.layers.ReLU()(h)

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d = min(self.dim * 2 ** (n_upsamplings - 2 - i), self.dim * 8)
            x = Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.relu(x)  # or h = keras.layers.ReLU()(h)

        x = Conv2DTranspose(3, 4, strides=2, padding='same')(x)
        x = tf.tanh(x)  # or h = keras.layers.Activation('tanh')(h)

        self.generator = Model(inputs=inputs, outputs=x, name=name)

    def get_norm_mode(self):
        if self.penalty_mode == 'none':
            d_norm = 'batch_norm'
        elif self.penalty_mode in ['dragan', 'wgan-gp']:  # cannot use batch normalization with gradient penalty
            # TODO(Lynn)
            # Layer normalization is more stable than instance normalization here,
            # but instance normalization works in other implementations.
            # Please tell me if you find out the cause.
            d_norm = 'layer_norm'
        return d_norm

    def build_discriminator(self, name='ConvDiscriminator'):

        n_down_samplings = int(np.log2(self.image_size) - np.log2(self.input_size))

        norm = self.get_norm_mode()
        Norm = get_norm_layer(norm)

        # 0
        h = inputs = Input(shape=(self.image_size, self.image_size, 3))

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        h = Conv2D(self.dim, 4, strides=2, padding='same')(h)
        h = tf.nn.leaky_relu(h, alpha=0.2)  # or keras.layers.LeakyReLU(alpha=0.2)(h)

        for i in range(n_down_samplings - 1):
            d = min(self.dim * 2 ** (i + 1), self.dim * 8)
            h = Conv2D(d, 4, strides=2, padding='same', use_bias=False)(h)
            h = Norm()(h)
            h = tf.nn.leaky_relu(h, alpha=0.2)  # or h = keras.layers.LeakyReLU(alpha=0.2)(h)

        if self.cgan:
            label_input = Input((1,))
            output_units = h.shape[1] * h.shape[2] * h.shape[3]
            label_embedding = Flatten()(Embedding(self.class_number, output_units)(label_input))
            label_embedding = Reshape((h.shape[1], h.shape[2], h.shape[3]))(label_embedding)
            inputs = [inputs, label_input]
            h = multiply([h, label_embedding])

        # 2: logistic
        h = Conv2D(1, self.input_size, strides=1, padding='valid')(h)

        self.discriminator = Model(inputs=inputs, outputs=h, name=name)

    def compile(self, init_rate_d, init_rate_g, loss_mode='wgan'):
        self.g_optimizer = SGD(learning_rate=init_rate_g)
        self.d_optimizer = SGD(learning_rate=init_rate_d)

        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(loss_mode)

    def train_epoch(self, batch_num, train_gen, *args, **kwargs):
        start_time = time.time()

        train_res_names = ['real_patch_d', 'real_img_d',
                           'fake_patch_d', 'fake_img_d',
                           'patch_g', 'img_g']
        train_res = np.zeros((batch_num, 6))

        q = Queue(10)

        producer = Thread(target=self.producer, args=(q, batch_num, train_gen))
        consumer = Thread(target=self.consumer, args=(q, batch_num))
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()

        train_res = train_res.mean(axis=0)
        print('Train time: {0}'.format(time.time() - start_time))
        log = 'Train - '
        for idx, name in enumerate(train_res_names):
            log += '{0}: {1} - '.format(name, train_res[idx])
        print(log)

        return train_res, train_res_names

    def test_model(self, test_dir):
        z = tf.random.normal(shape=(self.batch_size, self.noise_units))
        x_fake = self.generator(z, training=True)
        img = im.immerge(x_fake, n_rows=10).squeeze()
        im.imwrite(img, os.path.join(test_dir, 'iter-%09d.jpg' % self.g_optimizer.iterations.numpy()))
        return

    def consumer(self, q, batch_num):

        for _ in tqdm.tqdm(range(batch_num), desc='Inner Epoch Loop', total=batch_num):
            train = q.get()
            if train is None:
                break
            d_loss_dict = self.train_D(train)
            tl.summary(d_loss_dict, step=self.d_optimizer.iterations, name='D_losses')

            if self.d_optimizer.iterations.numpy() % 1 == 0:
                g_loss_dict = self.train_G()
                tl.summary(g_loss_dict, step=self.g_optimizer.iterations, name='G_losses')

            # sample
            if self.g_optimizer.iterations.numpy() % 100 == 0:
                self.test_model(r'F:\Code\Computer Science\depthGAN\test_images')

    @tf.function
    def train_G(self):
        with tf.GradientTape() as t:

            z = tf.random.normal(shape=(self.batch_size, self.noise_units))
            x_fake = self.generator(z, training=True)
            x_fake_d_logistic = self.discriminator(x_fake, training=True)
            g_loss = self.g_loss_fn(x_fake_d_logistic)

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return {'g_loss': g_loss}

    @tf.function
    def train_D(self, x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batch_size, self.noise_units))
            x_fake = self.generator(z, training=True)
            x_real_d_logistic = self.discriminator(x_real, training=True)
            x_fake_d_logistic = self.discriminator(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logistic, x_fake_d_logistic)
            gp = gradient_penalty(functools.partial(self.discriminator, training=True), x_real, x_fake,
                                  mode=self.penalty_mode)

            d_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.penalty_weight

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    @staticmethod
    def producer(q, batch_num, gen):
        for j in range(batch_num):
            x = next(gen)
            q.put(x)
        q.put([None, None])