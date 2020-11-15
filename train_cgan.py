from models.cGAN import *
from pipeline.coco_parser import *
from config import *
from pipeline.cifar import *
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Train cGAN')

parser.add_argument('--num_epoch', default=60)
parser.add_argument('--num_class', default=10)
parser.add_argument('--batch_size', default=32)
parser.add_argument('--test_per_cls', default=5)

parser.add_argument('--noise_units', default=128)
parser.add_argument('--image_size', default=32)
parser.add_argument('--input_size', default=4)
parser.add_argument('--dim', default=64)

parser.add_argument('--loss_mode', default='gan')
parser.add_argument('--penalty_mode', default='none')
parser.add_argument('--penalty_weight', default=10)
parser.add_argument('--g_per_d', default=1)
parser.add_argument('--cgan', default=False)
parser.add_argument('--patch', default=False)

parser.add_argument('--experiment_name', default='all_esig')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--category', default='vehicle')
parser.add_argument('--clear', default=False)

args = parser.parse_args()
# output_dir
if not args.experiment_name == 'none':
    experiment_name = args.experiment_name
else:
    experiment_name = ''
experiment_name += args.dataset + '_'
if args.dataset == 'COCO':
    for cat in args.category:
        experiment_name += cat
        experiment_name += '_'
experiment_name += args.loss_mode
if args.penalty_mode != 'none':
    experiment_name += '_%s' % args.penalty_mode
args.experiment_name = experiment_name

output_dir = os.path.join('output', args.experiment_name)

if args.clear:
    os.system("rm -rf " + output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'settings.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

tf.config.experimental_run_functions_eagerly(False)

if args.dataset == 'COCO':
    gen = COCOParser(coco_dir, resize=(64, 64), batch_size=args.batch_size)
    gen.set_super_category(args.category)
    categories = gen.categories
    train_gen = gen.balanced_gen('gan')
    args.num_class = len(categories)
else:
    train_gen = cifar_10_gen(args.batch_size)

cgan = ConditionalGAN(noise_unit=args.noise_units,
                      input_size=args.input_size,
                      image_size=args.image_size,
                      dim=args.dim,
                      class_number=args.num_class,
                      acgan=args.cgan,
                      batch_size=args.batch_size,
                      loss_mode=args.loss_mode,
                      penalty_mode=args.penalty_mode,
                      penalty_weight=args.penalty_weight,
                      patch=args.patch)

cgan.build_generator(name='G')
cgan.build_discriminator(name='D')

cgan.compile(0.0001, 0.0001)
cgan.set_ckpt(output_dir)
cur_epoch = cgan.cur_epoch
for epoch in range(args.num_epoch):
    if epoch < cur_epoch:
        continue
    print('epoch {0} / {1}'.format(epoch, args.num_epoch))
    cgan.train_epoch(batch_num=2000, train_gen=train_gen, g_per_d=args.g_per_d)
    cgan.test_model(output_dir, args.test_per_cls)
    cgan.save_ckpt(epoch)

cgan.test_batch(output_dir, 32)
