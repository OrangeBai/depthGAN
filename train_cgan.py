from models.cGAN import *
from pipeline.coco_parser import *
from config import *
from pipeline.cifar import *

import argparse

parser = argparse.ArgumentParser(description='Train cGAN')

parser.add_argument('--num_epoch', default=120)
parser.add_argument('--num_class', default=10)
parser.add_argument('--batch_size', default=32)


parser.add_argument('--noise_units', default=128)
parser.add_argument('--image_size', default=64)
parser.add_argument('--input_size', default=4)
parser.add_argument('--dim', default=64)

parser.add_argument('--loss_mode', default='wgan')
parser.add_argument('--penalty_mode', default='wgan-gp')
parser.add_argument('--penalty_weight', default=10)
parser.add_argument('--cgan', default=True)


args = parser.parse_args()

tf.config.experimental_run_functions_eagerly(False)


gen = COCOParser(coco_dir, resize=(64, 64), batch_size=args.batch_size)
gen.set_super_category('vehicle')
categories = gen.categories
train_gen = gen.balanced_gen('gan')

cgan = ConditionalGAN(noise_unit=args.noise_units,
                      input_size=args.input_size,
                      image_size=args.image_size,
                      dim=args.dim,
                      class_number=len(categories),
                      acgan=args.cgan,
                      batch_size=args.batch_size,
                      loss_mode=args.loss_mode,
                      penalty_mode=args.penalty_mode,
                      penalty_weight=args.penalty_weight)

a = cifar_10_gen(cgan=True)
b = next(a)

cgan.build_generator(name='G')
cgan.build_discriminator(name='D')

cgan.compile(0.0001, 0.0001)
for i in range(args.num_epoch):
    print('epoch {0} / {1}'.format(i, args.num_epoch))
    cgan.train_epoch(batch_num=1000, train_gen=train_gen)
