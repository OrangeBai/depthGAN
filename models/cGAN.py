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
from tensorflow.keras.losses import categorical_crossentropy


class ConditionalGAN:
    def __init__(self, noise_unit, input_size, image_size, dim, class_number, acgan, penalty_mode,
                 penalty_weight, batch_size, loss_mode):
        super().__init__()
        self.noise_units = noise_unit  # number of white noise units
        self.input_size = input_size  # size of generator convolutional input
        self.image_size = image_size  # image size
        self.dim = dim

        self.class_number = class_number
        self.batch_size = batch_size

        self.loss_mode = loss_mode
        self.d_loss_fn, self.g_loss_fn = get_adversarial_losses_fn(loss_mode)
        self.classifier_loss = categorical_crossentropy
        self.penalty_mode = penalty_mode
        self.penalty_weight = penalty_weight

        self.generator = None
        self.discriminator = None

        self.g_optimizer = None
        self.d_optimizer = None

        self.cgan = acgan

    def build_generator(self,
                        norm='batch_norm',
                        name='ConvGenerator'):

        Norm = get_norm_layer(norm)
        n_up_samplings = int(np.log2(self.image_size) - np.log2(self.input_size))

        h = inputs = Input(shape=(self.noise_units,))

        if self.cgan:
            label_input = Input((1,))
            label_embedding = Flatten()(Embedding(self.class_number, self.noise_units)(label_input))
            inputs = [inputs, label_input]
            h = multiply([h, label_embedding])

        h = Reshape((1, 1, self.noise_units))(h)
        # 1: 1x1 -> 4x4
        d = min(self.dim * 2 ** (n_up_samplings - 1), self.dim * 8)
        h = Conv2DTranspose(d, self.input_size, strides=1, padding='valid', use_bias=False)(h)
        h = Norm()(h)
        h = tf.nn.relu(h)  # or h = keras.layers.ReLU()(h)

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_up_samplings - 1):
            d = min(self.dim * 2 ** (n_up_samplings - 2 - i), self.dim * 8)
            h = Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
            h = Norm()(h)
            h = tf.nn.relu(h)  # or h = keras.layers.ReLU()(h)

            # h = Conv2DTranspose(d, 4, strides=1, padding='same', use_bias=False)(h)
            # h = Norm()(h)
            # h = tf.nn.relu(h)  # or h = keras.layers.ReLU()(h)

        h = Conv2DTranspose(3, 4, strides=2, padding='same')(h)
        h = tf.tanh(h)  # or h = keras.layers.Activation('tanh')(h)

        self.generator = Model(inputs=inputs, outputs=h, name=name)

    def get_norm_mode(self):
        d_norm = 'batch_norm'
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
        x = inputs = Input(shape=(self.image_size, self.image_size, 3))

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        x = Conv2D(self.dim, 4, strides=2, padding='same')(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)  # or keras.layers.LeakyReLU(alpha=0.2)(h)

        for i in range(n_down_samplings - 1):
            d = min(self.dim * 2 ** (i + 1), self.dim * 8)
            x = Conv2D(d, 4, strides=2, padding='same', use_bias=False)(x)
            x = Norm()(x)
            x = tf.nn.leaky_relu(x, alpha=0.2)  # or h = keras.layers.LeakyReLU(alpha=0.2)(h)

            # x = Conv2D(d, 4, strides=1, padding='same', use_bias=False)(x)
            # x = Norm()(x)
            # x = tf.nn.leaky_relu(x, alpha=0.2)  # or h = keras.layers.LeakyReLU(alpha=0.2)(h)

        # if self.cgan:
        #     label_input = Input((1,))
        #     output_units = x.shape[1] * x.shape[2] * x.shape[3]
        #     label_embedding = Flatten()(Embedding(self.class_number, output_units)(label_input))
        #     label_embedding = Reshape((x.shape[1], x.shape[2], x.shape[3]))(label_embedding)
        #     inputs = [inputs, label_input]
        #     x = multiply([x, label_embedding])

        # 2: logistic
        realness = Conv2D(1, self.input_size, strides=1, padding='valid')(x)
        if self.cgan:
            cls = Conv2D(self.class_number, self.input_size, padding='valid', activation=sigmoid)(x)
            cls = Flatten()(cls)
            out = [realness, cls]
        else:
            out = realness

        self.discriminator = Model(inputs=inputs, outputs=out, name=name)

    def compile(self, init_rate_d, init_rate_g):
        self.g_optimizer = Adam(init_rate_d, beta_1=0.5)
        self.d_optimizer = Adam(init_rate_g, beta_1=0.5)

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
        if self.cgan:
            z = tf.random.normal(shape=(self.class_number, self.noise_units))
            label = np.expand_dims(np.linspace(0, self.class_number - 1, self.class_number).astype('int'), axis=1)
            z = [z, label]
        else:
            z = tf.random.normal(shape=(100, self.noise_units))
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

            if self.d_optimizer.iterations.numpy() % 5 == 0:
                g_loss_dict = self.train_G(train)
                tl.summary(g_loss_dict, step=self.g_optimizer.iterations, name='G_losses')

            # sample
            if self.g_optimizer.iterations.numpy() % 100 == 0:
                self.test_model(r'F:\Code\Computer Science\depthGAN\test_images')

    @tf.function
    def train_G(self, x_real):
        real_image, label = x_real
        label = tf.keras.utils.to_categorical(label, self.class_number)
        with tf.GradientTape() as t:
            if self.cgan:
                x_fake = self.sample_fake(x_real[1], False)
            else:
                x_fake = self.sample_fake()
            pred_fake_res = self.discriminator(x_fake, training=True)

            if self.cgan:
                g_loss = self.g_loss_fn(pred_fake_res[0]) + self.classifier_loss(label, pred_fake_res[1])
            else:
                g_loss = self.g_loss_fn(pred_fake_res[0])

        g_grad = t.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))

        return {'g_loss': g_loss}

    @tf.function
    def train_D(self, x_real):
        real_image, label = x_real
        label = tf.keras.utils.to_categorical(label, self.class_number)
        with tf.GradientTape() as t:
            if self.cgan:
                x_fake = self.sample_fake(x_real[1], with_label=False)
            else:
                x_fake = self.sample_fake()

            pred_res_real = self.discriminator(real_image, training=True)
            pred_res_fake = self.discriminator(x_fake, training=True)

            if self.cgan:
                x_real_d_logistic, cls_real_d_sigmoid = pred_res_real
                x_fake_d_logistic, cls_fake_d_sigmoid = pred_res_fake
                cls_real_loss = self.classifier_loss(label, cls_real_d_sigmoid)
                cls_fake_loss = self.classifier_loss(label, cls_fake_d_sigmoid)
                d_loss = cls_real_loss + cls_fake_loss
            else:
                x_real_d_logistic = pred_res_real
                x_fake_d_logistic = pred_res_fake
                d_loss = tf.constant(0)

            x_real_d_loss, x_fake_d_loss = self.d_loss_fn(x_real_d_logistic, x_fake_d_logistic)

            gp = gradient_penalty(functools.partial(self.discriminator, training=True), real_image, x_fake,
                                  mode=self.penalty_mode, cgan=self.cgan)

            d_loss = d_loss + (x_real_d_loss + x_fake_d_loss) + gp * self.penalty_weight

        d_grad = t.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}

    @staticmethod
    def producer(q, batch_num, gen):
        for j in range(batch_num):
            x = next(gen)
            q.put(x)
        q.put([None, None])

    def sample_fake(self, real_label=None, with_label=False):
        if real_label is not None:
            z = tf.random.normal(shape=(self.batch_size, self.noise_units))
            x_fake = self.generator([z, real_label], training=True)
            if with_label:
                return [x_fake, real_label]
            else:
                return x_fake
        else:
            z = tf.random.normal(shape=(self.batch_size, self.noise_units))
            x_fake = self.generator(z, training=True)
            return x_fake
