from models.cGAN import *
from pipeline.coco_parser import *
from config import *
from pipeline.cifar import *

tf.config.experimental_run_functions_eagerly(False)
num_epoch = 60
num_class = 10
batch_size = 50
noise_size = 100

gen = COCOParser(coco_dir, resize=(64, 64), batch_size=batch_size)
# gen.set_super_category('vehicle')
categories = gen.categories
train_gen = gen.balanced_gen('gan')

cgan = ConditionalGAN((2, 2, 512), (32, 32, 3), num_class, batch_size=batch_size)
cgan.build_model()
cgan.compile(0.0002, 0.001)

cifar10_gen = cifar_10_gen()
for epoch in range(num_epoch):
    print('Train - {0} / {1}'.format(epoch, num_epoch))
    res = cgan.train_epoch(500, cifar10_gen)
    cgan.update_lr(num_epoch, epoch)
    cgan.test_model(test_dir, epoch, gen.categories)
    cgan.save_model(weights_dir, 'model_1')
cgan.load_model(weights_dir, 'model_1')
