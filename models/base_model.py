from tensorflow.keras.optimizers.schedules import *
from utils.lr_schedule import *
import time
from queue import Queue
from multiprocessing import Queue
from threading import Thread
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
import os


class BaseModel:
    """
    Base Model
    """

    def __init__(self):
        """
        Initialization method
        """
        self.generator = None  # Model
        self.discriminator = None
        self.model = None
        self.result = None  # Training Result
        self.reset_args = None  # Arguments for reset the model

        self._init_rate_d = None
        self._init_rate_g = None
        self._lr_schedule = static_learning_rate  # Learning rate schedule
        self._loss = None  # Loss

    def build_model(self, *args, **kwargs):
        self.reset_args = [args, kwargs]

    def compile(self, init_rate_d, init_rate_g, metrics=None, lr_schedule=static_learning_rate):
        """
        Compile method
        :param init_rate_d: Model loss
        :param init_rate_g: initial learning rate for combined model
        :param metrics: Metrics
        :param lr_schedule: Learning rate schedule function
        :return:
        """
        self._init_rate_d = init_rate_d
        self._init_rate_g = init_rate_g
        self._lr_schedule = lr_schedule

        # set learning rate
        learning_rate_fn_d = InverseTimeDecay(init_rate_d, 1, decay_rate=1e-5)
        optimizer_d = Adam()
        optimizer_d.learning_rate = learning_rate_fn_d
        self.discriminator.compile(optimizer=optimizer_d, loss=binary_crossentropy, metrics=metrics)

        learning_rate_fn_g = InverseTimeDecay(init_rate_g, 1, decay_rate=1e-5)
        optimizer_g = Adam()
        optimizer_g.learning_rate = learning_rate_fn_g
        self.model.compile(optimizer=optimizer_g, loss=binary_crossentropy, metrics=metrics)

    def train_epoch(self, batch_num, train_gen):
        start_time = time.time()
        metrics_name = self.model.metrics_names
        metrics_num = len(metrics_name)

        train_res = np.zeros((batch_num, metrics_num))

        q = Queue(10)

        producer = Thread(target=self.producer, args=(q, batch_num, train_gen))
        consumer = Thread(target=self.consumer, args=(q, batch_num, train_res))
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()

        train_res = train_res.mean(axis=0)
        print('Val time: {0}'.format(time.time() - start_time))
        log = 'Train - '
        for idx, name in enumerate(metrics_name):
            log += '{0}: {1} - '.format(name, train_res[idx])
        print(log)

        return train_res, metrics_name

    def update_lr(self, epoch_num, cur_epoch):
        new_lr_d = self._lr_schedule(epoch_num, cur_epoch, self._init_rate_d)
        if new_lr_d:
            learning_rate_d = InverseTimeDecay(new_lr_d, 1, decay_rate=1e-4)
            self.discriminator.optimizer.learning_rate = learning_rate_d
            print(new_lr_d)

        new_lr_g = self._lr_schedule(epoch_num, cur_epoch, self._init_rate_g)
        if new_lr_g:
            learning_rate_g = InverseTimeDecay(new_lr_g, 1, decay_rate=1e-4)
            self.model.optimizer.learning_rate = learning_rate_g
            print(new_lr_d)

    def multiple_training(self, exp_num):
        pass

    def reset_model(self):
        self.build_model()

    def save_model(self, weights_dir, name):
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
        path_d = os.path.join(weights_dir, name + '_d.h5')
        path_g = os.path.join(weights_dir, name + '_g.h5')
        self.generator.save_weights(path_g)
        self.discriminator.save_weights(path_d)
        return

    def load_model(self, weights_dir, name):
        path_d = os.path.join(weights_dir, name + '_d.h5')
        path_g = os.path.join(weights_dir, name + '_g.h5')

        self.generator.load_weights(path_g, by_name=True)
        self.discriminator.load_weights(path_d, by_name=True)
        return

    @staticmethod
    def producer(q, batch_num, gen):
        for j in range(batch_num):
            x = next(gen)
            q.put(x)
        q.put([None, None])

    def consumer(self, q, batch_num, train_res):
        for j in range(batch_num):
            x_train, y_train = q.get()
            if x_train is None:
                break
            train_res[j, :] = self.model.train_on_batch(np.array(x_train), np.array(y_train))
