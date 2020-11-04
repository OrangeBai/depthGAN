from models.nets import *
from models.base_model import *
from models.backbone import *
from numpy.random import random, randint
import cv2


class ConditionalGAN(GANBaseModel):
    def __init__(self, input_shape, image_shape, class_number, batch_size=32):
        super().__init__()
        self.input_shape = input_shape
        self.dense_units = int(np.prod(self.input_shape))
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

        noise_tensor = Input((self.dense_units // 2,))
        label_tensor = Input((self.class_number,))

        fake_image = self.generator([noise_tensor, label_tensor])
        validity = self.discriminator([fake_image, label_tensor])
        self.model = Model([noise_tensor, label_tensor], validity)

    def build_generator(self):
        dense_units = int(np.prod(self.input_shape))
        noise_tensor = Input((dense_units // 2,))
        label_tensor = Input((self.class_number,))
        combined_input = Concatenate(axis=1)([noise_tensor, label_tensor])

        x = dense_layer(combined_input, dense_units)

        x = Reshape(self.input_shape)(x)
        x = fast_up_projection(x, 128)
        x = bn_block(x, 128, (1, 1), 'LeakyReLU', 'block1_1', identity_block=False)
        x = bn_block(x, 128, (1, 1), 'LeakyReLU', 'block1_2', identity_block=True)

        x = fast_up_projection(x, 256)
        x = bn_block(x, 256, (1, 1), 'LeakyReLU', 'block2_1', identity_block=False)
        x = bn_block(x, 256, (1, 1), 'LeakyReLU', 'block2_2', identity_block=True)

        x = fast_up_projection(x, 256)
        x = conv_transpose_layer(x, 512, (3, 3))
        x = conv_transpose_layer(x, 512, (3, 3))

        x = fast_up_projection(x, 256)
        x = conv_transpose_layer(x, 256, (3, 3))
        x = conv_transpose_layer(x, 256, (3, 3))

        x = conv_transpose_layer(x, 128, (3, 3))
        x = conv_transpose_layer(x, 128, (3, 3))

        x = conv_transpose_layer(x, 64, (3, 3))
        x = conv_transpose_layer(x, 64, (3, 3))

        x = conv_transpose_layer(x, 3, (3, 3), batch_norm=False, activation=tanh)
        self.generator = Model([noise_tensor, label_tensor], x)
        return

    def build_discriminator(self):
        input_tensor = Input(self.image_shape)
        label_tensor = Input((self.class_number,))
        x = res20(input_tensor)
        x = conv_layer(x, 64, (3, 3))
        # output_patch = conv_layer(x, 1, (3, 3), activation=sigmoid)
        x = Flatten()(x)
        x = Concatenate()([x, label_tensor])
        x = dense_layer(x, 512)
        output_real = dense_layer(x, 1, sigmoid, batch_norm=False)

        self.discriminator = Model([input_tensor, label_tensor], output_real)
        return

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
        noise = np.random.random((self.class_number, self.dense_units // 2))
        fake_categories = [i for i in range(self.class_number)]
        fake_labels = tf.keras.utils.to_categorical(fake_categories, num_classes=self.class_number)
        fake_images = self.generator.predict_on_batch([noise, fake_labels])
        for idx, image in enumerate(fake_images):
            path = os.path.join(test_dir, 'epo_{0}_cat_{1}.jpg'.format(epoch_num, categories[idx]))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)
        return

    def consumer(self, q, batch_num, train_res):
        for j in range(batch_num):
            x_train, y_train = q.get()
            if x_train is None:
                break
            self.discriminator.trainable = True

            batch_size = x_train.shape[0]
            if batch_size > self.batch_size:
                batch_size = self.batch_size
            real_patch = 0.75 + 0.25 * np.random.random((batch_size, 8, 8, 1))
            real_gt = 0.75 + 0.25 * np.random.random((batch_size, 1))
            # real_loss = self.discriminator.train_on_batch([x_train[:batch_size], y_train[:batch_size]],
            #                                               [real_patch, real_gt])
            real_loss = self.discriminator.train_on_batch([x_train[:batch_size], y_train[:batch_size]], real_gt)

            noise = np.random.random((batch_size, self.dense_units // 2))
            fake_categories = randint(0, self.class_number, batch_size)
            fake_labels = tf.keras.utils.to_categorical(fake_categories, num_classes=self.class_number)
            fake_images = self.generator.predict_on_batch([noise, fake_labels])

            fake_gt = 0.25 * np.random.random((batch_size, 1))
            fake_patch = 0.25 * np.random.random((batch_size, 8, 8, 1))
            # fake_loss = self.discriminator.train_on_batch([fake_images, fake_labels], [fake_patch, fake_gt])
            fake_loss = self.discriminator.train_on_batch([fake_images, fake_labels], fake_gt)

            # train generator
            self.discriminator.trainable = False

            noise = np.random.random((batch_size, self.dense_units // 2))
            fake_categories = randint(0, self.class_number, batch_size)
            fake_labels = tf.keras.utils.to_categorical(fake_categories, num_classes=self.class_number)
            real_patch = 0.75 + 0.25 * np.random.random((batch_size, 8, 8, 1))
            real_gt = 0.75 + 0.25 * np.random.random((batch_size, 1))

            # g_loss = self.model.train_on_batch([noise, fake_labels], [real_patch, real_gt])

            g_loss = self.model.train_on_batch([noise, fake_labels], real_gt)

            # train_res[j, :] = [real_loss[1], real_loss[2], fake_loss[1], fake_loss[2], g_loss[1], g_loss[2]]
            train_res[j, :] = [0, real_loss, 0, fake_loss, 0, g_loss]
