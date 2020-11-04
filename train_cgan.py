from models.cGAN import *
from pipeline.coco_parser import *
from config import *

num_epoch = 60
num_class = 80

gen = COCOParser(coco_dir, resize=(64, 64), batch_size=32)
categories = gen.categories
train_gen = gen.balanced_gen('gan')

cgan = ConditionalGAN((2, 2, 512), (64, 64, 3), num_class)
cgan.build_model()
cgan.compile(0.0002, 0.001)

for i in range(num_epoch):
    print('Epoch {0} / {1}'.format(i, num_epoch))
    res = cgan.train_epoch(500, train_gen)
    cgan.update_lr(num_epoch, i)
    cgan.test_model(test_dir, i, categories)

    cgan.save_model(weights_dir, 'model_1')
# cgan.load_model(weights_dir, 'model_1')

print(1)
