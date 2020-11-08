from models.nets import *
from models.base_model import *
from models.backbone import *
from numpy.random import random, randint
from utils.losses import *
import cv2


class ConditionalGAN(GANBaseModel):
    def __init__(self, input_shape, image_shape, class_number, batch_size=32):
        super().__init__()
        self.input_shape = input_shape
        self.noise_units = 100
        self.dense_units = int(np.prod(self.input_shape))
        self.image_units = image_shape[0] * image_shape[1]
        self.image_shape = image_shape
        self.class_number = class_number
        self.batch_size = batch_size

    def build_model(self, *args, **kwargs):
        """
        Build conditional GAN model
        :param input_shape: size of noise, (width, height)
        :param image_shape: number of total class
        :param class_number:
        :param args:
        :param kwargs:
        :return:
        """
        self.build_generator()
        self.build_discriminator()

        noise_tensor = Input((self.noise_units,))
        label_tensor = Input((1,))

        fake_image = self.generator([noise_tensor, label_tensor])
        validity = self.discriminator([fake_image, label_tensor])
        self.model = Model([noise_tensor, label_tensor], validity)

    def build_generator(self):
        noise_tensor = Input((self.noise_units,))
        label_tensor = Input((1,))

        label_embedding = Flatten()(Embedding(self.class_number, self.noise_units)(label_tensor))

        combined_input = multiply([noise_tensor, label_embedding])

        x = Dense(self.dense_units)(combined_input)

        x = Activation(relu)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Reshape(self.input_shape)(x)

        # x = Conv2DTranspose(512, (5, 5), padding='same', strides=2)(x)
        # x = LeakyReLU(0.1)(x)

        x = Conv2DTranspose(256, (5, 5), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation(relu)(x)

        x = Conv2DTranspose(128, (5, 5), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation(relu)(x)

        x = Conv2DTranspose(64, (5, 5), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation(relu)(x)

        x = Conv2DTranspose(64, (5, 5), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation(relu)(x)

        x = Conv2DTranspose(3, (5, 5), padding='same')(x)
        img = Activation('tanh')(x)

        self.generator = Model([noise_tensor, label_tensor], img)
        self.generator.summary(160)
        return

    def build_discriminator(self):
        img = Input(shape=self.image_shape)

        x = GaussianNoise(0.05)(img)

        x = Conv2D(64, (3, 3), padding='same', strides=2)(x)
        # x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(0.2)(x)

        x = Conv2D(256, (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(0.2)(x)

        # x, patch = res14(x)

        patch_output = Conv2D(1, (3, 3), padding='same')(x)

        x = Conv2D(512, (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(0.2)(x)

        flat_img = Flatten()(x)

        label = Input(shape=(1,))
        label_embedding = Embedding(self.class_number, flat_img.shape[-1])(label)

        model_input = multiply([flat_img, label_embedding])

        nn = Dropout(0.4)(model_input)

        validity = Dense(1, activation='sigmoid')(nn)

        self.discriminator = Model([img, label], [validity, patch_output])
        self.discriminator.summary(160)

        return

    def compile(self, init_rate_d, init_rate_g, lr_schedule=static_learning_rate, metrics=None, *args, **kwargs):
        self._init_rate_d = init_rate_d
        self._init_rate_g = init_rate_g
        self._lr_schedule = lr_schedule

        optimizer_d = Adam(init_rate_d, beta_1=0.5)
        learning_rate_fn = InverseTimeDecay(init_rate_d, 1, decay_rate=0)
        optimizer_d.learning_rate = learning_rate_fn
        self.discriminator.compile(loss=binary_crossentropy, optimizer=optimizer_d)

        self.discriminator.trainable = False

        optimizer_g = Adam(init_rate_d, beta_1=0.5)
        learning_rate_fn = InverseTimeDecay(init_rate_g, 1, decay_rate=0)
        optimizer_g.learning_rate = learning_rate_fn
        self.model.compile(loss=binary_crossentropy, optimizer=optimizer_g)

    def train_epoch(self, batch_num, train_gen, *args, **kwargs):
        start_time = time.time()

        train_res_names = ['real_patch_d', 'real_img_d',
                           'fake_patch_d', 'fake_img_d',
                           'patch_g', 'img_g']
        train_res = np.zeros((batch_num, 6))

        q = Queue(10)

        producer = Thread(target=self.producer, args=(q, batch_num, train_gen))
        consumer = Thread(target=self.consumer, args=(q, batch_num, train_res))
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

    def test_model(self, test_dir, epoch_num, categories):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        noise = np.random.randn(self.class_number, self.noise_units)
        fake_categories = np.array([i for i in range(self.class_number)])

        fake_images = self.generator.predict_on_batch([noise, fake_categories])
        for idx, image in enumerate(fake_images):
            path = os.path.join(test_dir, 'epo_{0}_cat_{1}.jpg'.format(epoch_num, categories[idx]))
            cv2.imwrite(path, 127.5 * (image + 1))
        return

    def consumer(self, q, batch_num, train_res):
        for j in range(batch_num):
            x_train, y_train = q.get()
            if x_train is None:
                break

            batch_size = x_train.shape[0]
            if batch_size > self.batch_size:
                batch_size = self.batch_size

            self.discriminator.trainable = True

            real_gt = 0.9 + 0.1 * np.random.random((batch_size, 1))
            fake_gt = 0.1 * np.random.random((batch_size, 1))

            real_patch = 0.9 + 0.1 * np.random.random((batch_size, 4, 4))
            fake_patch = 0.1 * np.random.random((batch_size, 4, 4))
            if np.random.random() < 0.05:
                real_gt = np.zeros((batch_size, 1)) + (np.random.random() * 0.1)
                fake_gt = np.ones((batch_size, 1)) - (np.random.random() * 0.1)

            x = x_train[:batch_size]
            y = y_train[:batch_size]
            noise = np.random.randn(batch_size, self.noise_units)
            fake_images = self.generator.predict_on_batch([noise, y])

            real_loss = self.discriminator.train_on_batch([x, y], [real_gt, real_patch])
            fake_loss = self.discriminator.train_on_batch([fake_images, y], [fake_gt, fake_patch])

            # train generator
            self.discriminator.trainable = False

            noise = np.random.randn(batch_size, self.noise_units)
            valid = np.ones((batch_size, 1))
            valid_patch = np.ones((batch_size, 4, 4))
            g_loss = self.model.train_on_batch([noise, y], [valid, valid_patch])

            train_res[j, :] = [real_loss[1], real_loss[2], fake_loss[1], fake_loss[2], g_loss[1], g_loss[2]]
