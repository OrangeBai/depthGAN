from models.cGAN import *
from pipeline.coco_parser import *
from config import *

num_epoch = 60
num_class = 80

gen = COCOParser(coco_dir, resize=(64, 64), batch_size=32)
train_gen = gen.balanced_gen('gan')

cgan = ConditionalGAN((4, 4, 64), (64, 64, 3), num_class)
cgan.build_model()
cgan.compile(0.001, 0.001)

for i in range(num_epoch):
    print('Epoch {0} / {1}'.format(i, num_epoch))
    res = cgan.train_epoch(50, train_gen)
    cgan.update_lr(num_epoch, i)

cgan.save_model(weights_dir, 'model_1')
cgan.load_model(weights_dir, 'model_1')
noise = np.random.random((32, 512))
fake_categories = randint(0, num_class, 32)
fake_labels = tf.keras.utils.to_categorical(fake_categories, num_classes=num_class)
fake_images = cgan.generator.predict_on_batch([noise, fake_labels])
print(1)
