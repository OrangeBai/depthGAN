from tensorflow.keras.optimizers.schedules import *
from utils.lr_schedule import *
import time
from queue import Queue
from multiprocessing import Queue
from threading import Thread
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import os


class BaseModel:
    def __init__(self):
        self.model = None
        self.reset_args = None
        self._lr_schedule = static_learning_rate  # Learning rate schedule

    def build_model(self, *args, **kwargs):
        self.reset_args = [args, kwargs]

    def train_epoch(self, batch_num, train_gen, *args, **kwargs):
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
        print('Train time: {0}'.format(time.time() - start_time))
        log = 'Train - '
        for idx, name in enumerate(metrics_name):
            log += '{0}: {1} - '.format(name, train_res[idx])
        print(log)

        return train_res, metrics_name

    @staticmethod
    def producer(q, batch_num, gen):
        for j in range(batch_num):
            x = next(gen)
            q.put(x)
        q.put([None, None])

    def consumer(self, q, batch_num, train_res):
        pass

    def save_model(self, weights_dir, name):
        pass

    def load_model(self, weights_dir, name):
        pass


class GANBaseModel(BaseModel):
    """
    Base Model
    """

    def __init__(self):
        """
        Initialization method
        """
        super().__init__()
        self.generator = None  # Model
        self.discriminator = None

        self._init_rate_d = None
        self._init_rate_g = None
        self._loss = None  # Loss

    def train_epoch(self, batch_num, train_gen, *args, **kwargs):
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


