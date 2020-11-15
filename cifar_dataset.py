from pipeline.cifar import *
from config import *
import os
import cv2

gen = cifar_10_gen(32)


cifar_dataset = os.path.join(working_path, 'output', 'cifar10')
if not os.path.exists(cifar_dataset):
    os.makedirs(cifar_dataset)
for i in range(32):
    x, y = next(gen)
    for idx, img in enumerate(x):
        img_path = os.path.join(cifar_dataset, '{0}_{1}.jpg'.format(i, idx))
        img = (img.numpy() + 1) * 127.5
        cv2.imwrite(img_path, img)
