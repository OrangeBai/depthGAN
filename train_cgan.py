from models.cGAN import *
from pipeline.coco_parser import *
from config import *
from pipeline.cifar import *

tf.config.experimental_run_functions_eagerly(True)
num_epoch = 60
num_class = 10
batch_size = 50
noise_size = 100

gen = COCOParser(coco_dir, resize=(64, 64), batch_size=batch_size)
# gen.set_super_category('vehicle')
categories = gen.categories
train_gen = gen.balanced_gen('gan')

cgan = ConditionalGAN(noise_unit=128, input_size=2, image_size=32, dim=64, class_number=10, cgan=False,
                      penalty_mode='wgan-gp')

a = cifar_10_gen(cgan=False)
b = next(a)

cgan.build_generator()
cgan.build_discriminator()
cgan.generator.summary(160)
cgan.discriminator.summary(160)

cgan.compile(0.002, 0.002, loss_mode='wgan')
for i in range(20):
    cgan.train_epoch(batch_num=500, train_gen=a)
